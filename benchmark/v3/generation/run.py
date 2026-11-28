"""
Generation eval: run the production chain on the frozen v3 questions and grade
every answer with an LLM judge.

    python -m benchmark.v3.generation.run --pilot 5      # 5 dev questions, cost projection
    python -m benchmark.v3.generation.run                # all dev questions (after the pilot)

What one question goes through:

1. The production chain, unchanged: ``build_rag_chain(...).invoke`` with no
   chat history. That is routing, hybrid retrieval with the cross-encoder
   reranker on the exact v3 store, ``_QA_TEMPLATE``, and the production model
   (gemini-3.5-flash-lite) with production settings. Only the Gemini client is
   swapped for a cached, cost-tracked one.
2. The judge (``judge_v1.md``): one JSON verdict per field (refused, correct,
   grounded, length_fit), plus fields computed in code (word count, the exact
   refusal sentence, refusal_correct, wrongly_refused).

Pilot (``--pilot N``):
* N dev questions, one per question type, in a fixed order.
* Each answer is graded by flash-lite and by the stronger Flash candidate
  (gemini-3.7-flash, else gemini-3.5-flash if 3.7 is not available on the key).
* Prints the measured tokens and cost per call, and projects the total eval
  spend of the whole v4 plan for each judge.
* The stronger judge is chosen if its projection is under $1.50; otherwise
  flash-lite.

Full run (no ``--pilot``):
* Every dev question, graded by the judge the pilot chose.
* Noise re-runs: 20 answers re-judged, and 20 answers regenerated and judged.
* Writes the judge-check sheet for hand labels.

Results files (``results/``) hold verdicts, counts, tokens and hashes, never
question, answer or chunk text. Prompts and answers stay in the local,
gitignored cache (benchmark/.cache/generation/), so a rerun costs nothing.

Holdout is refused unless ``--split holdout --final-look`` (v4 Phase 6 only).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

GENERATION_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(GENERATION_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(V3_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.v3.generation import judge as J
from benchmark.v3.generation import llm as L

RESULTS_DIR = os.path.join(GENERATION_DIR, "results")
STORE_DIR = os.path.join(PROJECT_ROOT, "chroma_bench_v3_gen")
JUDGE_CHECK_PATH = os.path.join(GENERATION_DIR, "judge_check_dev.yaml")

ANSWER_MODEL = "gemini-3.5-flash-lite"          # production default (Dockerfile RAG_MODEL)
LITE_JUDGE = "gemini-3.5-flash-lite"
STRONG_JUDGES = ("gemini-3.7-flash", "gemini-3.5-flash")   # first one the key can use
STRONG_JUDGE_LIMIT_USD = 1.50

PILOT_QTYPES = ("exact_identifier", "paraphrase", "list_all", "multi_hop", "unanswerable",
                "keyword", "misspelled", "repo_identity")
NOISE_N = 20
NOISE_SEED = 20261128
JUDGE_CHECK_N = 20
JUDGE_CHECK_QUOTA = {"unanswerable": 3, "list_all": 2, "misspelled": 2, "repo_identity": 1,
                     "multi_hop": 3, "paraphrase": 4, "keyword": 3, "exact_identifier": 2}

# Calls the rest of the v4 plan makes, by kind (docs/ROADMAP_v4.md, cost table).
# Phase 2 counts include the pilot's 5 answers (cached by then): a slight overestimate.
PLAN_CALLS = {
    "phase 2 dev run":        {"answer": 127 + 20, "judge": 127 + 20 + 20},
    "phase 3 LLM rewrite":    {"rewrite": 127},
    "phase 4 list/small-doc": {"answer": 50, "judge": 50},
    "phase 5 length prompt":  {"answer": 127, "judge": 127},
    "phase 6 holdout x2":     {"answer": 108, "judge": 108},
}
# A first-turn rewrite prompt is small; priced from typical token counts.
REWRITE_TOKENS = (350, 40)
# Small-document answers (phase 4) carry whole documents; their extra input
# is covered by pricing phase 4 answers at this multiple of a measured answer.
PHASE4_ANSWER_FACTOR = 3.0


# ── question selection ──────────────────────────────────────────────────────

def pilot_questions(queries: list, n: int) -> list:
    """One per question type in PILOT_QTYPES order (first id of each), up to n."""
    picked = []
    for qtype in PILOT_QTYPES:
        of_type = sorted((q for q in queries if q["qtype"] == qtype), key=lambda q: q["id"])
        if of_type:
            picked.append(of_type[0])
        if len(picked) == n:
            break
    return picked


def seeded_sample(queries: list, n: int, seed: int) -> list:
    ids = sorted(q["id"] for q in queries)
    chosen = set(random.Random(seed).sample(ids, min(n, len(ids))))
    return [q for q in queries if q["id"] in chosen]


def judge_check_questions(queries: list) -> list:
    """20 dev questions stratified by type (JUDGE_CHECK_QUOTA), seeded."""
    out = []
    for qtype, k in JUDGE_CHECK_QUOTA.items():
        of_type = [q for q in queries if q["qtype"] == qtype]
        out.extend(seeded_sample(of_type, k, NOISE_SEED + len(qtype)))
    return sorted(out, key=lambda q: q["id"])


# ── one question ────────────────────────────────────────────────────────────

class Context:
    """Everything a question needs: data, gold, store, LLM, judge template."""

    def __init__(self, *, llm: L.CachedLLM, chroma_dir: str, parents: dict,
                 answer_model: str, template: str, build_chain=None,
                 gold_index: dict | None = None, positions: dict | None = None):
        from benchmark.v3 import gold as G

        self.G = G
        # Corpus-level gold (near-duplicate copies included), as the retrieval
        # eval scores it: {query id: (chunk index -> parts, n_parts)} and
        # {chunk key: chunk index}. Without them, direct span overlap is used.
        self.gold_index = gold_index
        self.positions = positions
        self.llm = llm
        self.chroma_dir = chroma_dir
        self.parents = parents
        self.answer_model = answer_model
        self.template = template
        if build_chain is None:
            from src.rag_chain import build_rag_chain as build_chain
        self.build_chain = build_chain


def answer_question(ctx: Context, q: dict, sample: int = 0) -> dict:
    """Run the production chain; returns the answer, its context and call stats."""
    replies: list = []
    llm_fn = L.bind(ctx.llm, ctx.answer_model, "answer", sample=sample, on_reply=replies.append)
    chain = ctx.build_chain("eval-uses-injected-llm", ctx.answer_model, ctx.chroma_dir, llm=llm_fn)
    started = time.monotonic()
    result = chain.invoke({"input": q["question"], "chat_history": []})
    docs = result.get("context") or []
    return {
        "answer": result["answer"],
        "docs": docs,
        "context_texts": [d.page_content for d in docs],
        "stats": {
            "calls": len(replies),
            "cached": all(r.cached for r in replies) if replies else True,
            "prompt_tokens": sum(r.usage.prompt for r in replies),
            "output_tokens": sum(r.usage.output for r in replies),
            "thought_tokens": sum(r.usage.thoughts for r in replies),
            "llm_seconds": round(sum(r.seconds for r in replies), 3),
            "chain_seconds": round(time.monotonic() - started, 3),
            "n_context": len(docs),
        },
    }


def query_shape(question: str) -> str:
    """"bare" for keyword-style input (3 words or fewer, no "?"), else "sentence"."""
    text = question.strip()
    return "bare" if len(text.split()) <= 3 and not text.endswith("?") else "sentence"


def gold_in_context(ctx: Context, q: dict, docs: list) -> dict:
    """
    Where the gold sat in what the model was given (no text):

    * gold_rank: rank of the first gold chunk among the context chunks
      (near-duplicate copies count, as in the retrieval eval);
    * parts_in_context / n_parts: how many answer parts had a gold chunk there;
    * gold_in_context: at least one gold chunk was there.

    A refusal with a gold chunk in front of the model is an answer-step
    failure; without one, a retrieval failure. An evidence string occurring
    in some other chunk does not count: short identifiers appear all over a
    code base. None for unanswerable questions.
    """
    if q["qtype"] == "unanswerable":
        return {"gold_rank": None, "parts_in_context": None, "n_parts": None,
                "gold_in_context": None}
    G = ctx.G
    if ctx.gold_index is not None and ctx.positions is not None:
        from src.hybrid_retrieval import chunk_key

        expanded, n_parts = ctx.gold_index[q["id"]]
        hits = [expanded.get(ctx.positions.get(chunk_key(d))) for d in docs]
    else:
        spans = G.resolve_gold(q, ctx.parents)
        n_parts = G.n_parts(spans)
        direct = G.direct_gold(spans, docs)
        hits = [direct.get(i) for i in range(len(docs))]
    ranks = [i + 1 for i, h in enumerate(hits) if h]
    parts = {p for h in hits if h for p in h}
    return {"gold_rank": ranks[0] if ranks else None, "parts_in_context": len(parts),
            "n_parts": n_parts, "gold_in_context": bool(ranks)}


def judge_answer(ctx: Context, q: dict, answer: str, context_texts: list, judge_model: str,
                 sample: int = 0) -> dict:
    """One judge verdict (plus derived fields), or an error marker."""
    answerable = q["qtype"] != "unanswerable"
    spans = ctx.G.resolve_gold(q, ctx.parents) if answerable else []
    prompt = J.build_prompt(
        ctx.template, question=q["question"], answerable=answerable,
        gold=J.format_gold(spans, ctx.parents, ctx.G.parent_slice),
        context=J.format_context(context_texts), answer=answer)
    last_error = None
    for attempt in (sample, sample + 1000):      # one retry on malformed output
        reply = ctx.llm(judge_model, "judge", prompt, config=J.JUDGE_CONFIG, sample=attempt)
        try:
            verdict = J.parse(reply.text)
        except J.JudgeError as exc:
            last_error = str(exc)
            continue
        return {**J.derive(verdict, answerable, answer), "judge_model": judge_model,
                "judge_tokens": {"prompt": reply.usage.prompt, "output": reply.usage.output,
                                 "thoughts": reply.usage.thoughts},
                "judge_seconds": round(reply.seconds, 3)}
    return {"error": last_error, "judge_model": judge_model}


# ── projection and judge choice ─────────────────────────────────────────────

def per_call_usd(rows: list, model: str, prices: dict) -> float | None:
    """Mean list-price cost of the given calls, from their token counts."""
    costs = [L.Usage(r["prompt"], r["output"], r["thoughts"]).cost(model, prices) for r in rows]
    return statistics.mean(costs) if costs else None


def project_plan(answer_usd: float, judge_usd: float, answer_model: str, prices: dict) -> dict:
    p_in, p_out = prices.get(answer_model, (0.0, 0.0))
    rewrite_usd = (REWRITE_TOKENS[0] * p_in + REWRITE_TOKENS[1] * p_out) / 1_000_000
    phases = {}
    for phase, calls in PLAN_CALLS.items():
        factor = PHASE4_ANSWER_FACTOR if phase.startswith("phase 4") else 1.0
        usd = (calls.get("answer", 0) * answer_usd * factor
               + calls.get("judge", 0) * judge_usd
               + calls.get("rewrite", 0) * rewrite_usd)
        phases[phase] = round(usd, 4)
    return {"phases": phases, "total_usd": round(sum(phases.values()), 4)}


def choose_judge(projections: dict, strong_model: str | None) -> tuple[str, str]:
    """The stronger judge if its whole-plan projection is under the limit."""
    if strong_model and projections.get(strong_model) is not None:
        total = projections[strong_model]["total_usd"]
        if total < STRONG_JUDGE_LIMIT_USD:
            return strong_model, (f"{strong_model} projects ${total:.2f} for the whole plan, "
                                  f"under ${STRONG_JUDGE_LIMIT_USD:.2f}")
        return LITE_JUDGE, (f"{strong_model} projects ${total:.2f}, not under "
                            f"${STRONG_JUDGE_LIMIT_USD:.2f}; using {LITE_JUDGE}")
    return LITE_JUDGE, "no stronger Flash model was available on this key"


# ── aggregates (no text) ────────────────────────────────────────────────────

def _score(value: str) -> float:
    return {"yes": 1.0, "partial": 0.5, "no": 0.0}[value]


def aggregate(rows: dict, resamples: int = 10_000, seed: int = 20261123) -> dict:
    """Per question type and overall: rates with bootstrap intervals."""
    from benchmark.v3 import stats as S

    groups: dict[str, list] = {"all": list(rows.values())}
    for r in rows.values():
        groups.setdefault(r["qtype"], []).append(r)
        if r.get("query_shape"):
            groups.setdefault(f"shape:{r['query_shape']}", []).append(r)
    out = {}
    for name, members in sorted(groups.items()):
        ok = [r for r in members if "error" not in r["verdict"]]
        graded = [r["verdict"] for r in ok]
        answerable = [r["verdict"] for r in ok if r["qtype"] != "unanswerable"]
        unanswerable = [r["verdict"] for r in ok if r["qtype"] == "unanswerable"]
        metrics = {}

        def add(metric, values, metrics=metrics):
            if values:
                idx = S.resample_matrix(len(values), resamples, seed)
                metrics[metric] = S.mean_ci(values, idx)

        add("correct_yes", [1.0 if v["correct"] == "yes" else 0.0 for v in answerable])
        add("correct_score", [_score(v["correct"]) for v in answerable])
        add("grounded_yes", [1.0 if v["grounded"] == "yes" else 0.0
                             for v in graded if v["grounded"] != "na"])
        add("wrongly_refused", [1.0 if v["wrongly_refused"] else 0.0 for v in answerable])
        add("refusal_correct", [1.0 if v["refusal_correct"] == "yes" else 0.0 for v in unanswerable])
        add("length_right", [1.0 if v["length_fit"] == "right" else 0.0 for v in graded])
        add("length_too_long", [1.0 if v["length_fit"] == "too_long" else 0.0 for v in graded])
        add("length_too_short", [1.0 if v["length_fit"] == "too_short" else 0.0 for v in graded])
        words = [v["word_count"] for v in graded]
        out[name] = {"n": len(members), "graded": len(graded), "metrics": metrics,
                     "median_words": statistics.median(words) if words else None}
    return out


def refusal_diagnosis(rows: dict) -> dict:
    """
    Answerable questions the answer refused, by where the gold was:
    all parts in context, some parts, or none (a retrieval miss).
    """
    out = {"all_parts_in_context": [], "some_parts_in_context": [], "gold_not_in_context": []}
    for qid, r in sorted(rows.items()):
        v = r.get("verdict", {})
        if r["qtype"] == "unanswerable" or not v.get("wrongly_refused"):
            continue
        g = r.get("gold_context") or {}
        if not g.get("gold_in_context"):
            out["gold_not_in_context"].append(qid)
        elif g.get("parts_in_context", 0) < (g.get("n_parts") or 0):
            out["some_parts_in_context"].append(qid)
        else:
            out["all_parts_in_context"].append(qid)
    return out


def agreement(pairs: list[tuple[dict, dict]]) -> dict:
    """Share of identical verdicts per field between two gradings."""
    out = {}
    for f in ("refused", "correct", "grounded", "length_fit"):
        same = [a[f] == b[f] for a, b in pairs if "error" not in a and "error" not in b]
        out[f] = {"agree": sum(same), "n": len(same)}
    return out


# ── reporting to the console (no text) ──────────────────────────────────────

def _row(cells: list, widths: list) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def print_pilot(rows: dict, judges: list, ledger: dict, projections: dict, chosen: tuple,
                usd_per_call: dict, unavailable: list) -> None:
    print("\nPILOT: per question (no text is printed)")
    widths = [9, 17, 6, 8] + [26] * len(judges)
    print(_row(["id", "qtype", "words", "context"] + judges, widths))
    for qid, r in rows.items():
        cells = [qid, r["qtype"], r["answer_stats"].get("word_count", ""),
                 r["answer_stats"]["n_context"]]
        for jm in judges:
            v = r["verdicts"].get(jm, {})
            cells.append("error" if "error" in v else
                         f"c={v['correct']} g={v['grounded']} r={v['refused']} l={v['length_fit']}")
        print(_row(cells, widths))
    print("\nCOST PER CALL (list price, from measured tokens)")
    for model_purpose, usd in usd_per_call.items():
        print(f"  {model_purpose:40s} ${usd:.6f}" if usd is not None else
              f"  {model_purpose:40s} n/a")
    print("\nPROJECTED TOTAL EVAL SPEND FOR THE WHOLE v4 PLAN, BY JUDGE")
    for jm, proj in projections.items():
        if proj is None:
            print(f"  {jm:28s} " + ("not available on this key" if jm in unavailable
                                    else "not measured (an earlier candidate was available)"))
            continue
        print(f"  {jm:28s} ${proj['total_usd']:.2f}  " +
              ", ".join(f"{k}: ${v:.3f}" for k, v in proj["phases"].items()))
    print(f"\nJUDGE CHOSEN: {chosen[0]}  ({chosen[1]})")
    print(f"THIS RUN: {ledger['new_calls']} new calls, {ledger['cached_calls']} cached, "
          f"${ledger['new_usd']:.4f} spent")


def print_full(report: dict) -> None:
    agg = report["aggregates"]
    print(f"\nDEV RESULTS (judge: {report['judge_model']}; n = {agg['all']['n']})")
    widths = [16, 4, 13, 13, 13, 13, 13, 8]
    print(_row(["qtype", "n", "correct_yes", "correct_score", "grounded", "refusal_ok",
                "length_right", "words"], widths))

    def pct(m, k):
        x = m.get(k)
        return "-" if x is None else f"{100 * x['mean']:.0f} [{100 * x['ci_low']:.0f},{100 * x['ci_high']:.0f}]"
    for name, g in agg.items():
        m = g["metrics"]
        print(_row([name, g["n"], pct(m, "correct_yes"), pct(m, "correct_score"),
                    pct(m, "grounded_yes"), pct(m, "refusal_correct"), pct(m, "length_right"),
                    g["median_words"]], widths))
    d = report["wrong_refusals"]
    print(f"\nWRONG REFUSALS of answerable questions: {sum(len(v) for v in d.values())}")
    for kind, ids in d.items():
        print(f"  {kind.replace('_', ' ')}: {len(ids)} ({', '.join(ids) or 'none'})")
    n = report["noise"]
    print("\nNOISE (same answer re-judged / answer regenerated then judged): agreement per field")
    for kind in ("rejudge", "regenerate"):
        print(f"  {kind:11s} " + ", ".join(f"{f} {v['agree']}/{v['n']}" for f, v in n[kind].items()))
    led = report["ledger"]
    print(f"\nTHIS RUN: {led['new_calls']} new calls, {led['cached_calls']} cached, "
          f"${led['new_usd']:.4f} spent")


# ── setup ───────────────────────────────────────────────────────────────────

def api_key() -> str | None:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    except ImportError:
        pass
    return (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip() or None


def prepare_store(chunks: list, store: str | None) -> str:
    """The exact v3 store, built once and reused (it is deterministic)."""
    from benchmark.v3.evaluate import build_store
    from src.vector_store import count_chunks

    path = store or STORE_DIR
    if os.path.isdir(path):
        n = count_chunks(path)
        if n != len(chunks):
            raise SystemExit(f"{path} holds {n} chunks, expected {len(chunks)}: delete that "
                             "folder and rerun to rebuild it.")
        print(f"[gen-eval] reusing the eval store ({n} chunks)", flush=True)
        return path
    print(f"[gen-eval] building the exact eval store ({len(chunks)} chunks, a few minutes "
          "the first time)", flush=True)
    return build_store(chunks, path, exact=True)


def require_reranker() -> None:
    from src import reranker as R

    model = R.get_reranker()
    if model is None:
        raise SystemExit("the reranker did not load, so this would not be the production "
                         "path. Check that RAG_RERANKER is not 'off' and the model downloads.")
    model.score("warm-up", ["warm-up passage"], budget_s=120.0)   # first call is slow


def answer_prompt_hash() -> str:
    import hashlib

    from src import rag_chain
    blob = (f"{rag_chain._QA_TEMPLATE}\0{rag_chain._CONVERSATIONAL_SYSTEM_PROMPT}"
            f"\0{rag_chain._CONTEXTUALIZE_TEMPLATE}")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def parse_prices(items: list[str] | None) -> dict:
    prices = dict(L.PRICES)
    for item in items or []:
        model, _, pair = item.partition("=")
        p_in, _, p_out = pair.partition(",")
        prices[model.strip()] = (float(p_in), float(p_out))
    return prices


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1, ensure_ascii=True)
        handle.write("\n")


# ── runs ────────────────────────────────────────────────────────────────────

def run_pilot(ctx: Context, questions: list, prices: dict) -> dict:
    rows: dict = {}
    calls: dict[str, list] = {}
    strong_available = None
    unavailable: list[str] = []
    judges = [LITE_JUDGE]
    for q in questions:
        a = answer_question(ctx, q)
        a["stats"]["word_count"] = J.word_count(a["answer"])
        calls.setdefault(f"{ctx.answer_model}|answer", []).append(
            {"prompt": a["stats"]["prompt_tokens"], "output": a["stats"]["output_tokens"],
             "thoughts": a["stats"]["thought_tokens"]})
        verdicts = {LITE_JUDGE: judge_answer(ctx, q, a["answer"], a["context_texts"], LITE_JUDGE)}
        for strong in ([strong_available] if strong_available else STRONG_JUDGES):
            try:
                verdicts[strong] = judge_answer(ctx, q, a["answer"], a["context_texts"], strong)
                strong_available = strong
                break
            except L.CallFailed as exc:
                if exc.code in (400, 403, 404):
                    if strong not in unavailable:
                        unavailable.append(strong)
                    print(f"[gen-eval] {strong} is not available on this key (HTTP {exc.code}); "
                          "trying the next candidate", flush=True)
                    continue
                raise
        for jm, v in verdicts.items():
            if jm not in judges:
                judges.append(jm)
            if "judge_tokens" in v:
                t = v["judge_tokens"]
                calls.setdefault(f"{jm}|judge", []).append(
                    {"prompt": t["prompt"], "output": t["output"], "thoughts": t["thoughts"]})
        rows[q["id"]] = {"qtype": q["qtype"], "domain": q["domain"],
                         "query_shape": query_shape(q["question"]),
                         "gold_context": gold_in_context(ctx, q, a["docs"]),
                         "answer_stats": a["stats"], "verdicts": verdicts}
        print(f"[gen-eval] {q['id']} done", flush=True)

    usd_per_call = {k: per_call_usd(v, k.split("|")[0], prices) for k, v in calls.items()}
    answer_usd = usd_per_call.get(f"{ctx.answer_model}|answer") or 0.0
    projections = {}
    for jm in [LITE_JUDGE, *STRONG_JUDGES]:
        judge_usd = usd_per_call.get(f"{jm}|judge")
        projections[jm] = (None if judge_usd is None else
                           project_plan(answer_usd, judge_usd, ctx.answer_model, prices))
    chosen = choose_judge(projections, strong_available)
    return {"rows": rows, "judges": judges, "usd_per_call": usd_per_call,
            "projections": projections, "chosen": chosen, "strong_available": strong_available,
            "unavailable": unavailable}


def run_full(ctx: Context, questions: list, judge_model: str, noise_n: int) -> dict:
    rows: dict = {}
    answers: dict = {}
    for q in questions:
        a = answer_question(ctx, q)
        answers[q["id"]] = a
        verdict = judge_answer(ctx, q, a["answer"], a["context_texts"], judge_model)
        rows[q["id"]] = {"qtype": q["qtype"], "domain": q["domain"],
                         "query_shape": query_shape(q["question"]),
                         "gold_context": gold_in_context(ctx, q, a["docs"]),
                         "answer_stats": a["stats"], "verdict": verdict}
        print(f"[gen-eval] {q['id']} done", flush=True)

    noise_qs = seeded_sample(questions, noise_n, NOISE_SEED)
    rejudge, regenerate = [], []
    for q in noise_qs:
        a = answers[q["id"]]
        again = judge_answer(ctx, q, a["answer"], a["context_texts"], judge_model, sample=1)
        rejudge.append((rows[q["id"]]["verdict"], again))
        b = answer_question(ctx, q, sample=1)
        new = judge_answer(ctx, q, b["answer"], b["context_texts"], judge_model)
        regenerate.append((rows[q["id"]]["verdict"], new))
        print(f"[gen-eval] noise {q['id']} done", flush=True)
    return {"rows": rows, "answers": answers,
            "noise": {"ids": [q["id"] for q in noise_qs],
                      "rejudge": agreement(rejudge), "regenerate": agreement(regenerate)}}


def write_judge_check(ctx: Context, questions: list, answers: dict, path: str) -> list:
    """
    The hand-label sheet, as two local files (gitignored: they hold text):
    ``judge_check_dev.md`` to read (question, gold passages, retrieved context,
    answer) and ``judge_check_dev.yaml`` with an empty label block per item.
    The judge's own grades are deliberately not shown.
    """
    import yaml

    md_path = os.path.splitext(path)[0] + ".md"
    picked = judge_check_questions(questions)
    lines = ["# Judge check: 20 dev answers to label by hand", "",
             f"Write your labels in `{os.path.basename(path)}` (same folder), then run",
             "`python -m benchmark.v3.generation.agreement`. The rubric is `judge_v1.md`.",
             "This file contains document text: do not commit it.", ""]
    for q in picked:
        a = answers[q["id"]]
        answerable = q["qtype"] != "unanswerable"
        spans = ctx.G.resolve_gold(q, ctx.parents) if answerable else []
        lines += [f"## {q['id']}", "",
                  f"**Question:** {q['question']}", "",
                  f"**Answerable from the documents:** {'yes' if answerable else 'no'}", "",
                  "**Gold passages:**", "", "```text",
                  J.format_gold(spans, ctx.parents, ctx.G.parent_slice), "```", "",
                  "**Retrieved context (what the model was given):**", "", "```text",
                  J.format_context(a["context_texts"]), "```", "",
                  "**Answer:**", "", "```text", a["answer"], "```", ""]
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    header = (
        "# Judge check labels. Read judge_check_dev.md, then fill every field:\n"
        "#   refused:    yes | no\n"
        "#   correct:    yes | partial | no | na   (na when the question is not answerable)\n"
        "#   grounded:   yes | no | na             (na when the answer makes no factual claim)\n"
        "#   length_fit: too_short | right | too_long\n"
        "# Then run: python -m benchmark.v3.generation.agreement\n")
    items = [{"id": q["id"], "label": {"refused": "", "correct": "", "grounded": "",
                                       "length_fit": ""}} for q in picked]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump({"items": items}, handle, sort_keys=False)
    return [q["id"] for q in picked]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--final-look", action="store_true",
                        help="required with --split holdout (v4 Phase 6 only)")
    parser.add_argument("--pilot", type=int, default=0, metavar="N",
                        help="pilot on N dev questions with the cost projection")
    parser.add_argument("--judge", help="judge model for the full run (default: the pilot's choice)")
    parser.add_argument("--answer-model", default=ANSWER_MODEL)
    parser.add_argument("--max-usd", type=float, default=0.50, help="cap on new spend in this run")
    parser.add_argument("--rpm", type=float, default=10.0, help="requests per minute (throttle)")
    parser.add_argument("--noise", type=int, default=NOISE_N)
    parser.add_argument("--price", action="append", metavar="MODEL=IN,OUT",
                        help="override a price, USD per 1M tokens")
    parser.add_argument("--archive", help="local psf/requests tarball at the pinned commit")
    parser.add_argument("--store", help="eval store directory (default: chroma_bench_v3_gen)")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.split == "holdout" and not args.final_look:
        raise SystemExit("refusing to touch holdout: it is scored once, in v4 Phase 6, with "
                         "--split holdout --final-look.")
    if args.split == "holdout" and args.pilot:
        raise SystemExit("no pilot on holdout")

    from benchmark.v3 import dataset as D
    from benchmark.v3 import freeze
    from benchmark.v3.corpus import load_corpus

    manifest = freeze.verify()
    judge_sha = J.verify_judge()
    splits = freeze.load_split_assignments()
    prices = parse_prices(args.price)
    key = api_key()
    started = time.time()

    chunks, parents, _meta = load_corpus(args.archive)
    data = D.load()
    queries = [q for q in data["queries"] if splits[q["id"]] == args.split]
    chroma_dir = prepare_store(chunks, args.store)
    require_reranker()

    def estimate(model: str) -> float:
        p_in, p_out = prices.get(model, (0.0, 0.0))
        return (6000 * p_in + 1500 * p_out) / 1_000_000   # a generous single call

    llm = L.CachedLLM(L.gemini_backend(key) if key else None, L.Cache(),
                      max_usd=args.max_usd, rpm=args.rpm, prices=prices, estimate_usd=estimate)
    from benchmark.v3 import gold as G
    from src.hybrid_retrieval import chunk_key

    resolved, _errors = D.resolve_all(queries, chunks, parents, G.NearDuplicates(chunks))
    gold_index = {qid: (r.expanded, r.n_parts) for qid, r in resolved.items()}
    ctx = Context(llm=llm, chroma_dir=chroma_dir, parents=parents,
                  answer_model=args.answer_model, template=J.load_template(),
                  gold_index=gold_index, positions={chunk_key(c): i for i, c in enumerate(chunks)})
    common = {
        "split": args.split,
        "manifest_sha256": manifest["manifest_sha256"],
        "judge_version": J.JUDGE_VERSION,
        "judge_prompt_sha256": judge_sha,
        "answer_prompt_sha256": answer_prompt_hash(),
        "answer_model": args.answer_model,
        "prices_usd_per_1m": {m: list(p) for m, p in prices.items()},
    }

    try:
        if args.pilot:
            questions = pilot_questions(queries, args.pilot)
            print(f"[gen-eval] pilot: {len(questions)} dev questions; up to "
                  f"{len(questions) * 3} calls", flush=True)
            out = run_pilot(ctx, questions, prices)
            report = {**common, "kind": "pilot", "questions": [q["id"] for q in questions],
                      "per_question": out["rows"], "usd_per_call": out["usd_per_call"],
                      "projections": out["projections"],
                      "strong_judge_available": out["strong_available"],
                      "strong_judges_unavailable": out["unavailable"],
                      "judge_chosen": out["chosen"][0], "judge_choice_reason": out["chosen"][1],
                      "thinking_setting_dropped_for": sorted(llm.dropped_thinking),
                      "ledger": llm.ledger.summary(), "seconds": round(time.time() - started, 1)}
            path = os.path.join(RESULTS_DIR, f"pilot_{args.split}.json")
            write_json(path, report)
            print_pilot(out["rows"], out["judges"], report["ledger"], out["projections"],
                        out["chosen"], out["usd_per_call"], out["unavailable"])
        else:
            judge_model = args.judge
            if not judge_model:
                pilot_path = os.path.join(RESULTS_DIR, "pilot_dev.json")
                if not os.path.exists(pilot_path):
                    raise SystemExit("run the pilot first (--pilot 5), or pass --judge MODEL")
                with open(pilot_path, encoding="utf-8") as handle:
                    judge_model = json.load(handle)["judge_chosen"]
            out = run_full(ctx, queries, judge_model, args.noise)
            report = {**common, "kind": "full", "judge_model": judge_model,
                      "per_question": out["rows"], "aggregates": aggregate(out["rows"]),
                      "wrong_refusals": refusal_diagnosis(out["rows"]),
                      "noise": out["noise"],
                      "thinking_setting_dropped_for": sorted(llm.dropped_thinking),
                      "ledger": llm.ledger.summary(), "seconds": round(time.time() - started, 1)}
            if args.split == "dev":
                report["judge_check_ids"] = write_judge_check(ctx, queries, out["answers"],
                                                              JUDGE_CHECK_PATH)
            path = os.path.join(RESULTS_DIR, f"{args.split}.json")
            write_json(path, report)
            print_full(report)
            if args.split == "dev":
                print("\nJudge check (local, do not commit): read "
                      f"{os.path.relpath(os.path.splitext(JUDGE_CHECK_PATH)[0] + '.md', PROJECT_ROOT)}"
                      f", label in {os.path.relpath(JUDGE_CHECK_PATH, PROJECT_ROOT)}")
    except L.SpendCapReached as exc:
        print(f"\nSTOPPED: {exc}. Finished calls are cached; rerun with a higher --max-usd "
              "to continue from where it stopped.")
        return 2
    except L.CallFailed as exc:
        print(f"\nSTOPPED: {exc}. Finished calls are cached; rerunning the same command "
              "resumes from where it stopped.")
        return 3
    print(f"\nwritten: {os.path.relpath(path, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
