"""
Text-free features of cached answers, for diagnosing a prompt change.

    python -m benchmark.v3.generation.answer_features --ids mh-05,mh-08,mh-09

Replays the production chain for the given dev questions under both answer
prompts: the Phase 2 baseline prompt (kept below, verified against
results/dev.json's hash) and the current one. It reads every answer from the
local cache. It makes no API calls: it runs without a key, so an uncached
answer is reported as missing instead of being generated. Cost: $0.

Prints, per question and prompt, only booleans and counts, never text:
* words: answer length;
* refusal_exact: the exact refusal sentence is present;
* says_missing: the answer states that something is not covered, mentioned,
  specified or provided;
* parts_in_answer: how many gold parts have their key fact (the evidence
  string) in the answer, against the number of parts;
* parts_in_context: how many gold parts had a gold chunk in the context.

says_missing with every part in context is the case to look for: the model
declaring a part missing that it was given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

GENERATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(GENERATION_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.v3.generation import llm as L
from benchmark.v3.generation import run as R

# _QA_TEMPLATE as it was for the Phase 2 baseline (main at e21360a).
BASELINE_QA_TEMPLATE = """You are a helpful assistant that answers questions strictly from the provided documents.

Use ONLY the context below to answer the question.
If the context does not contain the answer, say exactly:
"I don't have enough information in the provided documents to answer this."
Do not fabricate any information.

Exception: if the question is actually casual conversation (a greeting, thanks,
acknowledgment, or small talk) rather than a genuine request for document
information, ignore the context below and respond naturally and warmly instead.
{history_block}
Context from retrieved documents:
{context}

Question: {question}

Answer:"""

MISSING = re.compile(
    r"\b(?:do|does|did)\s*(?:not|n't)\s+(?:cover|mention|specify|say|include|contain|provide|"
    r"state|describe|explain|detail|address|discuss|indicate)"
    r"|\bnot\s+(?:covered|mentioned|specified|provided|stated|included|described|explained|"
    r"detailed|addressed|discussed|available)\b"
    r"|\bno\s+(?:information|details|mention)\b"
    r"|\b(?:isn't|is not|aren't|are not)\s+(?:covered|mentioned|specified|provided|stated)\b",
    re.IGNORECASE)


def prompt_hash(qa_template: str) -> str:
    from src import rag_chain

    blob = (f"{qa_template}\0{rag_chain._CONVERSATIONAL_SYSTEM_PROMPT}"
            f"\0{rag_chain._CONTEXTUALIZE_TEMPLATE}")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def features(answer: str, spans: list, gold_context: dict, norm) -> dict:
    answer_norm = norm(answer)
    parts: dict[str, bool] = {}
    for s in spans:
        for p in s.parts:
            hit = bool(s.evidence) and norm(s.evidence) in answer_norm
            parts[p] = parts.get(p, False) or hit
    return {
        "words": len(answer.split()),
        "refusal_exact": R.J.is_exact_refusal(answer),
        "says_missing": bool(MISSING.search(answer)),
        "parts_in_answer": f"{sum(parts.values())}/{len(parts)}" if parts else "-",
        "parts_in_context": (f"{gold_context.get('parts_in_context')}/{gold_context.get('n_parts')}"
                             if gold_context.get("n_parts") else "-"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", required=True, help="comma-separated dev question ids")
    parser.add_argument("--archive", help="local psf/requests tarball at the pinned commit")
    parser.add_argument("--out", help="also write the features as JSON (no text)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from benchmark.v3 import dataset as D
    from benchmark.v3 import freeze
    from benchmark.v3 import gold as G
    from benchmark.v3.corpus import load_corpus
    from src import rag_chain
    from src.hybrid_retrieval import chunk_key

    with open(os.path.join(R.RESULTS_DIR, "dev.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    if prompt_hash(BASELINE_QA_TEMPLATE) != base["answer_prompt_sha256"]:
        raise SystemExit("the kept baseline prompt does not match results/dev.json's hash")

    freeze.verify()
    splits = freeze.load_split_assignments()
    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    bad = [i for i in ids if splits.get(i) != "dev"]
    if bad:
        raise SystemExit(f"not dev questions: {', '.join(bad)}")

    chunks, parents, _ = load_corpus(args.archive)
    queries = {q["id"]: q for q in D.load()["queries"] if q["id"] in ids}
    store = R.prepare_store(chunks, None)
    R.require_reranker()
    resolved, _ = D.resolve_all(list(queries.values()), chunks, parents, G.NearDuplicates(chunks))
    llm = L.CachedLLM(None, L.Cache(), max_usd=0.0, rpm=0)      # cache only: no key, no calls
    ctx = R.Context(llm=llm, chroma_dir=store, parents=parents, answer_model=R.ANSWER_MODEL,
                    template="", gold_index={k: (r.expanded, r.n_parts) for k, r in resolved.items()},
                    positions={chunk_key(c): i for i, c in enumerate(chunks)})

    current = rag_chain._QA_TEMPLATE
    out: dict = {}
    print(f"{'id':8s} {'prompt':9s} {'words':>5s} {'refusal':>8s} {'says_missing':>12s} "
          f"{'parts_in_answer':>15s} {'parts_in_context':>16s}")
    for qid in ids:
        q = queries[qid]
        spans = G.resolve_gold(q, parents) if q["qtype"] != "unanswerable" else []
        for label, template in (("baseline", BASELINE_QA_TEMPLATE), ("current", current)):
            rag_chain._QA_TEMPLATE = template
            try:
                a = R.answer_question(ctx, q)
            except L.CallFailed:
                print(f"{qid:8s} {label:9s} (answer not in the cache)")
                continue
            finally:
                rag_chain._QA_TEMPLATE = current
            f = features(a["answer"], spans, R.gold_in_context(ctx, q, a["docs"]), G._norm)
            out.setdefault(qid, {})[label] = f
            print(f"{qid:8s} {label:9s} {f['words']:5d} {f['refusal_exact']!s:>8s} "
                  f"{f['says_missing']!s:>12s} {f['parts_in_answer']:>15s} "
                  f"{f['parts_in_context']:>16s}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=1)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
