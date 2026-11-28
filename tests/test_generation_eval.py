"""
Generation eval harness (benchmark/v3/generation), offline.

A fake model stands in for Gemini everywhere: nothing reaches the network and
no key is needed. Covers the cache (a rerun makes no calls), the spend cap,
retries and the thinking-setting fallback, judge parsing and derived fields,
the cost projection and the judge-model rule, aggregation, judge-vs-human
agreement, a pilot end to end through the real judge prompt, and that results
files never contain question, answer or chunk text.
"""

import json
import os

import pytest

from benchmark.v3.generation import agreement as A
from benchmark.v3.generation import judge as J
from benchmark.v3.generation import llm as L
from benchmark.v3.generation import run as R

GOOD_VERDICT = {"refused": "no", "correct": "yes", "grounded": "yes",
                "unsupported_claims": 0, "length_fit": "right", "notes": "fine"}
SECRET_QUESTION = "What is the zebra-quokka redirect limit?"
SECRET_ANSWER = "The zebra-quokka limit is 30 redirects."
SECRET_CHUNK = "zebra-quokka chunk text: the limit is 30 redirects."


class FakeBackend:
    """Answers with SECRET_ANSWER, judges with GOOD_VERDICT; records calls."""

    def __init__(self, verdict=None, fail=None):
        self.calls = []
        self.verdict = verdict or GOOD_VERDICT
        self.fail = fail or {}          # model -> exception to raise

    def __call__(self, model, prompt, config):
        self.calls.append((model, config is not None))
        if model in self.fail:
            raise self.fail[model]
        if config:                       # judge calls carry the JSON config
            return json.dumps(self.verdict), L.Usage(2000, 80, 40)
        return SECRET_ANSWER, L.Usage(1500, 120, 0)


class HttpError(Exception):
    def __init__(self, code, message="error"):
        super().__init__(message)
        self.code = code


def _llm(tmp_path, backend, **kw):
    kw.setdefault("max_usd", 1.0)
    kw.setdefault("rpm", 0)
    return L.CachedLLM(backend, L.Cache(str(tmp_path / "cache.jsonl")), sleep=lambda s: None, **kw)


# ── cache, cap, retries ─────────────────────────────────────────────────────

def test_a_repeated_call_is_served_from_the_cache_even_in_a_new_process(tmp_path):
    backend = FakeBackend()
    first = _llm(tmp_path, backend)("m", "answer", "prompt")
    again = _llm(tmp_path, backend)("m", "answer", "prompt")      # fresh object, same file

    assert len(backend.calls) == 1
    assert not first.cached and again.cached
    assert again.text == first.text and again.usage == first.usage


def test_a_new_sample_bypasses_the_cache(tmp_path):
    backend = FakeBackend()
    llm = _llm(tmp_path, backend)
    llm("m", "answer", "prompt")
    llm("m", "answer", "prompt", sample=1)
    assert len(backend.calls) == 2


def test_cost_counts_thinking_tokens_as_output():
    usage = L.Usage(prompt=1_000_000, output=100_000, thoughts=100_000)
    assert usage.cost("gemini-3.5-flash-lite", L.PRICES) == pytest.approx(0.30 + 0.2 * 2.50)


def test_the_spend_cap_stops_before_a_call_that_would_exceed_it(tmp_path):
    backend = FakeBackend()
    llm = _llm(tmp_path, backend, max_usd=0.001, estimate_usd=lambda m: 0.002)
    with pytest.raises(L.SpendCapReached):
        llm("gemini-3.5-flash-lite", "answer", "prompt")
    assert backend.calls == []


def test_cached_calls_never_count_against_the_cap(tmp_path):
    backend = FakeBackend()
    _llm(tmp_path, backend)("m", "answer", "prompt")
    capped = _llm(tmp_path, backend, max_usd=0.0, estimate_usd=lambda m: 1.0)
    assert capped("m", "answer", "prompt").cached


def test_rate_limits_are_retried_then_reported(tmp_path):
    flaky = FakeBackend()
    attempts = {"n": 0}

    def backend(model, prompt, config):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise HttpError(429)
        return flaky(model, prompt, config)

    assert _llm(tmp_path, backend)("m", "answer", "p").text == SECRET_ANSWER
    always = FakeBackend(fail={"m": HttpError(429)})
    with pytest.raises(L.CallFailed) as err:
        _llm(tmp_path, always, retries=2)("m", "answer", "q")
    assert err.value.code == 429


def test_a_model_that_rejects_the_thinking_setting_is_called_without_it(tmp_path):
    seen = []

    def backend(model, prompt, config):
        seen.append(dict(config or {}))
        if config and "thinking_level" in config:
            raise HttpError(400, "thinking_level is not supported for this model")
        return json.dumps(GOOD_VERDICT), L.Usage(10, 10, 0)

    llm = _llm(tmp_path, backend)
    llm("m", "judge", "p", config={"temperature": 0, "thinking_level": "LOW"})
    assert "thinking_level" not in seen[-1] and llm.dropped_thinking == {"m"}


def test_without_a_key_an_uncached_call_fails_clearly(tmp_path):
    with pytest.raises(L.CallFailed, match="no API key"):
        _llm(tmp_path, None)("m", "answer", "p")


# ── judge ───────────────────────────────────────────────────────────────────

def test_the_pinned_judge_prompt_matches_the_file():
    assert J.verify_judge() == J.sha256_file(J.JUDGE_PATH)


def test_a_changed_judge_prompt_is_refused(tmp_path):
    changed = tmp_path / "judge_v1.md"
    changed.write_text(J.load_template() + "\nBe generous.\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed since it was pinned"):
        J.verify_judge(judge_path=str(changed))


def test_judge_output_is_validated():
    assert J.parse("```json\n" + json.dumps(GOOD_VERDICT) + "\n```")["correct"] == "yes"
    with pytest.raises(J.JudgeError):
        J.parse("not json")
    with pytest.raises(J.JudgeError):
        J.parse(json.dumps({**GOOD_VERDICT, "correct": "mostly"}))


def test_derived_fields_are_computed_in_code():
    refusal = J.REFUSAL_SENTENCE
    unans = J.derive({**GOOD_VERDICT, "refused": "yes", "correct": "yes"}, False, refusal)
    assert unans["correct"] == "na" and unans["refusal_correct"] == "yes"
    assert unans["refusal_exact"] and unans["wrongly_refused"] is None
    ans = J.derive({**GOOD_VERDICT, "refused": "yes"}, True, refusal)
    assert ans["wrongly_refused"] is True and ans["refusal_correct"] == "na"
    assert J.derive(GOOD_VERDICT, True, "one two three")["word_count"] == 3


def test_gold_is_grouped_by_part_with_its_key_fact():
    from benchmark.v3.gold import Span

    parents = {("doc.txt", 0): "intro " * 400 + "The limit is 30 redirects." + " outro" * 400}
    start = parents[("doc.txt", 0)].index("The limit")
    spans = [Span("doc.txt", 0, 0, len(parents[("doc.txt", 0)]), ("a",), evidence="30 redirects"),
             Span("doc.txt", 0, start, start + 26, ("b",), evidence="limit is 30")]
    text = J.format_gold(spans, parents, lambda p, s: p[(s.file, s.page)][s.start:s.end])
    assert text.startswith("PART 1") and "PART 2" in text
    assert "KEY FACT: 30 redirects" in text and "30 redirects" in text.split("PART 2")[0]
    assert len(text) < 2 * J.MAX_SPAN_CHARS            # the long span was cut around its fact


# ── projection and judge choice ─────────────────────────────────────────────

def test_the_stronger_judge_is_chosen_only_under_the_limit():
    cheap = {"gemini-3.7-flash": {"total_usd": 1.20}}
    dear = {"gemini-3.7-flash": {"total_usd": 1.80}}
    assert R.choose_judge(cheap, "gemini-3.7-flash")[0] == "gemini-3.7-flash"
    assert R.choose_judge(dear, "gemini-3.7-flash")[0] == R.LITE_JUDGE
    assert R.choose_judge({}, None)[0] == R.LITE_JUDGE


def test_the_plan_projection_adds_up():
    proj = R.project_plan(answer_usd=0.001, judge_usd=0.002, answer_model="gemini-3.5-flash-lite",
                          prices=L.PRICES)
    answers = sum(c.get("answer", 0) * (R.PHASE4_ANSWER_FACTOR if p.startswith("phase 4") else 1)
                  for p, c in R.PLAN_CALLS.items())
    judges = sum(c.get("judge", 0) for c in R.PLAN_CALLS.values())
    rewrites = 127 * (350 * 0.30 + 40 * 2.50) / 1e6
    assert proj["total_usd"] == pytest.approx(answers * 0.001 + judges * 0.002 + rewrites, abs=1e-3)


def test_pilot_questions_cover_one_of_each_type_in_a_fixed_order():
    qs = [{"id": f"{t[:3]}-{i}", "qtype": t} for t in R.PILOT_QTYPES for i in (2, 1)]
    picked = R.pilot_questions(qs, 5)
    assert [q["qtype"] for q in picked] == list(R.PILOT_QTYPES[:5])
    assert all(q["id"].endswith("-1") for q in picked)


# ── aggregation and agreement ───────────────────────────────────────────────

def _row(qtype, **verdict):
    v = J.derive({**GOOD_VERDICT, **verdict}, qtype != "unanswerable", "a b c")
    return {"qtype": qtype, "verdict": v}


def test_aggregates_per_type_skip_judge_errors():
    rows = {"a": _row("keyword"), "b": _row("keyword", correct="partial"),
            "c": _row("unanswerable", refused="yes"),
            "d": {"qtype": "keyword", "verdict": {"error": "bad json"}}}
    agg = R.aggregate(rows, resamples=200)
    assert agg["keyword"]["n"] == 3 and agg["keyword"]["graded"] == 2
    assert agg["keyword"]["metrics"]["correct_score"]["mean"] == pytest.approx(0.75)
    assert agg["unanswerable"]["metrics"]["refusal_correct"]["mean"] == 1.0
    assert agg["all"]["metrics"]["correct_yes"]["n"] == 2


def test_judge_human_agreement_reports_direction():
    human = {"q1": {"refused": "no", "correct": "partial", "grounded": "no", "length_fit": "right"},
             "q2": {"refused": "no", "correct": "yes", "grounded": "yes", "length_fit": "too_long"},
             "q3": {"refused": "no", "correct": "no", "grounded": "yes", "length_fit": "right"}}
    judged = {"q1": {"refused": "no", "correct": "yes", "grounded": "yes", "length_fit": "right"},
              "q2": {"refused": "no", "correct": "yes", "grounded": "yes", "length_fit": "right"},
              "q3": {"refused": "no", "correct": "no", "grounded": "no", "length_fit": "too_short"}}
    out = A.compare(human, judged)
    assert out["correct"]["agree"] == 2 and out["correct"]["judge_more_lenient"] == ["q1"]
    assert out["grounded"]["judge_more_lenient"] == ["q1"]
    assert out["grounded"]["judge_stricter"] == ["q3"]
    assert out["length_fit"]["judge_more_lenient"] == ["q2"]
    assert out["length_fit"]["judge_stricter"] == ["q3"]
    assert out["passes_bar"] is False


def test_cohen_kappa():
    assert A.cohen_kappa([("y", "y"), ("n", "n")]) == 1.0
    assert A.cohen_kappa([("y", "n"), ("n", "y")]) == pytest.approx(-1.0)
    assert A.cohen_kappa([]) is None


def test_unfinished_labels_are_rejected(tmp_path):
    sheet = tmp_path / "sheet.yaml"
    sheet.write_text("items:\n- id: q1\n  label: {refused: no, correct: '', grounded: yes, "
                     "length_fit: right}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="q1: correct"):
        A.load_labels(str(sheet))


# ── end to end: pilot and full run with a fake chain ────────────────────────

class _Doc:
    def __init__(self, text, start=7):
        self.page_content = text
        self.metadata = {"source": "doc.txt", "source_type": "upload", "start_index": start}


def _fake_build_chain(api_key, model, chroma_dir, llm=None):
    class Chain:
        def invoke(self, inputs):
            reply = llm(f"PROMPT for: {inputs['input']}")
            # A non-gold chunk first, then the gold one (it starts at offset 7).
            return {"answer": reply.text,
                    "context": [_Doc("unrelated text", start=500), _Doc(SECRET_CHUNK)]}
    return Chain()


def _ctx(tmp_path, backend):
    parents = {("doc.txt", 0): "Intro. " + SECRET_CHUNK + " End."}
    llm = _llm(tmp_path, backend)
    return R.Context(llm=llm, chroma_dir="unused", parents=parents,
                     answer_model=R.ANSWER_MODEL, template=J.load_template(),
                     build_chain=_fake_build_chain)


QUESTIONS = [
    {"id": "eid-01", "qtype": "exact_identifier", "domain": "code", "question": SECRET_QUESTION,
     "gold": [{"file": "doc.txt", "anchor": "the limit is 30 redirects", "evidence": "30"}]},
    {"id": "una-01", "qtype": "unanswerable", "domain": "docs",
     "question": SECRET_QUESTION + " (unanswerable)"},
]


def test_the_pilot_falls_back_to_the_next_stronger_judge_and_writes_no_text(tmp_path):
    backend = FakeBackend(fail={"gemini-3.7-flash": HttpError(404, "model not found")})
    out = R.run_pilot(_ctx(tmp_path, backend), QUESTIONS, L.PRICES)

    assert out["strong_available"] == "gemini-3.5-flash"
    assert set(out["rows"]["eid-01"]["verdicts"]) == {R.LITE_JUDGE, "gemini-3.5-flash"}
    assert out["rows"]["una-01"]["verdicts"][R.LITE_JUDGE]["refusal_correct"] == "no"
    assert out["projections"]["gemini-3.7-flash"] is None
    assert out["chosen"][0] in (R.LITE_JUDGE, "gemini-3.5-flash")

    path = tmp_path / "pilot.json"
    R.write_json(str(path), {"per_question": out["rows"], "projections": out["projections"]})
    written = path.read_text(encoding="utf-8")
    assert "zebra-quokka" not in written and "redirect" not in written


def test_a_rerun_of_the_full_eval_makes_no_new_calls(tmp_path):
    backend = FakeBackend()
    first = R.run_full(_ctx(tmp_path, backend), QUESTIONS, R.LITE_JUDGE, noise_n=2)
    calls = len(backend.calls)
    ctx = _ctx(tmp_path, backend)
    again = R.run_full(ctx, QUESTIONS, R.LITE_JUDGE, noise_n=2)

    assert len(backend.calls) == calls and ctx.llm.ledger.new_calls == 0
    verdicts = {q: {k: v for k, v in r["verdict"].items() if k != "judge_seconds"}
                for q, r in first["rows"].items()}
    assert {q: {k: v for k, v in r["verdict"].items() if k != "judge_seconds"}
            for q, r in again["rows"].items()} == verdicts
    assert all(r["answer_stats"]["cached"] for r in again["rows"].values())
    assert first["noise"]["rejudge"]["correct"]["n"] == 2
    assert "zebra" not in json.dumps({k: v for k, v in first.items() if k != "answers"})


def test_the_judge_check_sheet_holds_text_and_is_gitignored(tmp_path):
    pytest.importorskip("yaml")
    backend = FakeBackend()
    ctx = _ctx(tmp_path, backend)
    qs = [{**QUESTIONS[0], "id": f"kw-{i}", "qtype": "keyword"} for i in range(5)]
    out = R.run_full(ctx, qs, R.LITE_JUDGE, noise_n=0)
    sheet = tmp_path / "judge_check_dev.yaml"
    ids = R.write_judge_check(ctx, qs, out["answers"], str(sheet))

    assert len(ids) == 3                         # the keyword quota
    reading = (tmp_path / "judge_check_dev.md").read_text(encoding="utf-8")
    assert "zebra-quokka" in reading and reading.count("## kw-") == 3
    labels = sheet.read_text(encoding="utf-8")
    assert "zebra" not in labels and labels.count("id: kw-") == 3
    # The empty sheet is rejected until every field is filled in.
    with pytest.raises(SystemExit, match="unfinished or invalid labels"):
        A.load_labels(str(sheet))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, ".gitignore"), encoding="utf-8") as handle:
        ignored = handle.read()
    assert "benchmark/v3/generation/judge_check_*.yaml" in ignored
    assert "benchmark/v3/generation/judge_check_*.md" in ignored


def test_holdout_is_refused_without_the_final_look_flag():
    with pytest.raises(SystemExit, match="refusing to touch holdout"):
        R.main(["--split", "holdout"])


# ── the production chain takes an injected model ────────────────────────────

def test_build_rag_chain_uses_an_injected_llm_and_creates_no_client(monkeypatch):
    from google import genai

    from src.rag_chain import build_rag_chain

    def no_client(*args, **kwargs):
        raise AssertionError("a Gemini client was created")

    monkeypatch.setattr(genai, "Client", no_client)
    prompts = []

    class Reply:
        text = "Hello there."

    def fake(prompt):
        prompts.append(prompt)
        return Reply()

    chain = build_rag_chain("unused", "m", None, llm=fake)   # no documents: conversational route
    assert chain.invoke({"input": "hello", "chat_history": []})["answer"] == "Hello there."
    assert len(prompts) == 1


# ── Windows checkouts (CRLF) still verify ───────────────────────────────────

def test_frozen_and_judge_hashes_ignore_crlf_conversion(tmp_path):
    from benchmark.v3 import freeze

    lf, crlf = tmp_path / "lf.md", tmp_path / "crlf.md"
    lf.write_bytes(b"line one\nline two\n")
    crlf.write_bytes(b"line one\r\nline two\r\n")
    assert freeze.sha256_file(str(lf)) == freeze.sha256_file(str(crlf))
    assert J.sha256_file(str(lf)) == J.sha256_file(str(crlf))


def test_a_crlf_corpus_document_is_replaced_by_its_lf_copy(tmp_path, monkeypatch):
    import hashlib

    from benchmark import build_corpus as B

    original = b"planets\nmoons\n"
    doc = tmp_path / "notes.txt"
    doc.write_bytes(original.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(B, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setitem(B.EXPECTED_SHA256, "notes.txt", hashlib.sha256(original).hexdigest())

    copy = B._lf_copy_if_it_matches(str(doc), "notes.txt", "crlf-digest")
    with open(copy, "rb") as handle:
        assert handle.read() == original

    doc.write_bytes(b"something else\r\n")
    with pytest.raises(RuntimeError, match="sha256"):
        B._lf_copy_if_it_matches(str(doc), "notes.txt", "other-digest")


def test_rows_record_query_shape_and_where_the_gold_was_in_context(tmp_path):
    backend = FakeBackend(verdict={**GOOD_VERDICT, "refused": "yes", "correct": "no"})
    qs = [*QUESTIONS, {**QUESTIONS[0], "id": "eid-02", "question": "DEFAULT_LIMIT"}]
    out = R.run_full(_ctx(tmp_path, backend), qs, R.LITE_JUDGE, noise_n=0)
    rows = out["rows"]

    assert rows["eid-01"]["query_shape"] == "sentence"
    assert rows["eid-02"]["query_shape"] == "bare"
    assert rows["eid-01"]["gold_context"] == {"gold_rank": 2, "parts_in_context": 1,
                                              "n_parts": 1, "gold_in_context": True}
    assert rows["una-01"]["gold_context"]["gold_in_context"] is None
    # Answerable questions refused with the gold in front of the model are
    # answer-step failures; without it, retrieval failures.
    assert R.refusal_diagnosis(rows) == {"all_parts_in_context": ["eid-01", "eid-02"],
                                         "some_parts_in_context": [], "gold_not_in_context": []}
    agg = R.aggregate(rows, resamples=100)
    assert agg["shape:bare"]["n"] == 1 and agg["shape:sentence"]["n"] == 2


def test_an_evidence_string_outside_the_gold_chunk_does_not_count(tmp_path):
    """A short identifier appears in many chunks; only a gold chunk counts."""
    q = {**QUESTIONS[0], "id": "eid-09",
         "gold": [{"file": "doc.txt", "anchor": "the limit is 30 redirects", "evidence": "30"},
                  {"file": "doc.txt", "anchor": "End.", "part": "second"}]}
    ctx = _ctx(tmp_path, FakeBackend())
    elsewhere = _Doc("unrelated text that also says 30", start=500)
    assert R.gold_in_context(ctx, q, [elsewhere])["gold_in_context"] is False
    some = R.gold_in_context(ctx, q, [elsewhere, _Doc(SECRET_CHUNK)])
    assert some == {"gold_rank": 2, "parts_in_context": 1, "n_parts": 2, "gold_in_context": True}
    rows = {"eid-09": {"qtype": "exact_identifier", "gold_context": some,
                       "verdict": {"wrongly_refused": True}}}
    assert R.refusal_diagnosis(rows)["some_parts_in_context"] == ["eid-09"]


# ── Phase 2b: the answer prompt, the plan cap, the comparison ──────────────

def test_the_answer_prompt_keeps_the_exact_refusal_sentence_and_formats():
    from src import rag_chain

    assert J.REFUSAL_SENTENCE in rag_chain._QA_TEMPLATE
    prompt = rag_chain._QA_TEMPLATE.format(context="C", question="Q", history_block="")
    assert "term, name or identifier" in prompt and "answer every part" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_the_plan_cap_counts_every_cached_call(tmp_path):
    backend = FakeBackend()
    _llm(tmp_path, backend)("gemini-3.5-flash-lite", "answer", "earlier run")   # ~$0.00075
    llm = _llm(tmp_path, backend, max_total_usd=0.0009, estimate_usd=lambda m: 0.0005)
    assert llm.spent_before_usd == pytest.approx(L.Usage(1500, 120, 0).cost(
        "gemini-3.5-flash-lite", L.PRICES))
    with pytest.raises(L.SpendCapReached, match="plan's"):
        llm("gemini-3.5-flash-lite", "answer", "a new prompt")
    assert llm("gemini-3.5-flash-lite", "answer", "earlier run").cached   # cached is still free


def test_a_bad_tag_is_refused():
    with pytest.raises(SystemExit, match="--tag"):
        R.main(["--tag", "Not OK!"])


def _run(rows: dict, **extra) -> dict:
    per_question = {}
    for qid, (qtype, shape, verdict) in rows.items():
        v = J.derive({**GOOD_VERDICT, **verdict}, qtype != "unanswerable", "w " * 20)
        per_question[qid] = {"qtype": qtype, "query_shape": shape, "verdict": v}
    return {"split": "dev", "manifest_sha256": "m", "judge_model": "j", "judge_version": "v1",
            "judge_prompt_sha256": "h", "answer_prompt_sha256": extra.get("prompt", "p1"),
            "tag": extra.get("tag"), "per_question": per_question,
            "wrong_refusals": extra.get("wrong_refusals", {}),
            "noise": {"rejudge": {}, "regenerate": {}}}


REFUSED = {"refused": "yes", "correct": "no", "grounded": "na"}


def test_the_comparison_scores_the_phase_2b_criteria():
    from benchmark.v3.generation import compare as C

    base = _run({
        "eid-01": ("exact_identifier", "bare", REFUSED),          # answer-step refusal
        "crs-05": ("multi_hop", "sentence", REFUSED),             # some parts in context
        "isw-01": ("exact_identifier", "bare", REFUSED),          # no gold in context
        "par-05": ("paraphrase", "sentence", REFUSED),            # no gold in context
        "kw-01": ("keyword", "bare", {}),
        "una-01": ("unanswerable", "sentence", {"refused": "yes", "grounded": "na"}),
    }, wrong_refusals={"all_parts_in_context": ["eid-01"], "some_parts_in_context": ["crs-05"],
                       "gold_not_in_context": ["isw-01", "par-05"]})
    new = _run({
        "eid-01": ("exact_identifier", "bare", {}),               # now answered, correct
        "crs-05": ("multi_hop", "sentence", {"correct": "partial"}),
        "isw-01": ("exact_identifier", "bare", {"correct": "no", "grounded": "no"}),  # guessing
        "par-05": ("paraphrase", "sentence", REFUSED),
        "kw-01": ("keyword", "bare", {"correct": "partial"}),     # got worse
        "una-01": ("unanswerable", "sentence", {"refused": "yes", "grounded": "na"}),
    }, prompt="p2", tag="refusal")

    assert C.check_comparable(base, new) == []
    out = C.compare(base, new)
    crit = out["criteria"]
    assert crit["answer_step_refusals_remaining"]["remaining"] == []
    assert crit["answer_step_refusals_remaining"]["pass"]
    assert crit["unanswerable_refused"]["pass"]
    assert crit["correct_up"]["pass"]                 # 2 up (eid-01, crs-05) vs 1 down (kw-01)
    assert not crit["grounded"]["pass"]               # isw-01 is ungrounded: 1 of 5 graded
    assert not out["passes"]
    assert out["flips"]["correct_up"] == ["crs-05", "eid-01"]
    assert out["flips"]["correct_down"] == ["kw-01"]
    assert out["gold_not_in_context_after"]["isw-01"].endswith("(guessing)")
    assert out["gold_not_in_context_after"]["par-05"] == "still refused"
    assert out["by_group"]["shape:bare"]["n"] == 3


def test_runs_with_different_judges_are_not_compared():
    from benchmark.v3.generation import compare as C

    base = _run({"a": ("keyword", "bare", {})})
    other = {**_run({"a": ("keyword", "bare", {})}), "judge_model": "another"}
    with pytest.raises(SystemExit, match="judge_model"):
        C.check_comparable(base, other)
    assert C.check_comparable(base, _run({"a": ("keyword", "bare", {})})) == [
        "the answer prompt is the same in both runs"]


# ── answer features (text-free diagnosis of cached answers) ─────────────────

def test_the_kept_baseline_prompt_is_the_one_the_baseline_was_scored_with():
    from benchmark.v3.generation import answer_features as F

    results = os.path.join(os.path.dirname(J.__file__), "results", "dev.json")
    if not os.path.exists(results):
        pytest.skip("no baseline results in this checkout")
    with open(results, encoding="utf-8") as handle:
        assert F.prompt_hash(F.BASELINE_QA_TEMPLATE) == json.load(handle)["answer_prompt_sha256"]


@pytest.mark.parametrize("answer, missing", [
    ("The documents do not specify the timeout.", True),
    ("This part is not covered in the provided documents.", True),
    ("The context doesn't mention a default value.", True),
    ("There is no information about retries.", True),
    ("The default timeout is 30 seconds, set in adapters.py.", False),
    ("It does not retry by default; set max_retries to change that.", False),
])
def test_says_missing_detects_a_declared_gap(answer, missing):
    from benchmark.v3.generation import answer_features as F

    assert bool(F.MISSING.search(answer)) is missing


def test_features_count_parts_whose_key_fact_is_in_the_answer():
    from benchmark.v3.generation import answer_features as F
    from benchmark.v3.gold import Span, _norm

    spans = [Span("a.py", 0, 0, 10, ("x",), evidence="max_redirects"),
             Span("b.py", 0, 0, 10, ("y",), evidence="TooManyRedirects")]
    f = F.features("It raises TooManyRedirects; the limit is not specified here.", spans,
                   {"parts_in_context": 2, "n_parts": 2}, _norm)
    assert f["parts_in_answer"] == "1/2" and f["parts_in_context"] == "2/2"
    assert f["says_missing"] and not f["refusal_exact"]


# ── refusal2: multi-hop criterion, no-gold outcomes, three kept prompts ─────

def test_multi_hop_correct_yes_must_return_to_the_baseline_and_no_gold_outcomes_are_split():
    from benchmark.v3.generation import compare as C

    gold = {"gold_in_context": True}
    nogold = {"gold_in_context": False}

    def runs(mh_new_correct):
        base = _run({"mh-1": ("multi_hop", "sentence", {}),
                     "mh-2": ("multi_hop", "sentence", {"correct": "partial"}),
                     "ng-1": ("paraphrase", "sentence", REFUSED),
                     "ng-2": ("keyword", "bare", REFUSED)},
                    wrong_refusals={"all_parts_in_context": [], "some_parts_in_context": [],
                                    "gold_not_in_context": ["ng-1", "ng-2"]})
        new = _run({"mh-1": ("multi_hop", "sentence", {"correct": mh_new_correct}),
                    "mh-2": ("multi_hop", "sentence", {"correct": "partial"}),
                    "ng-1": ("paraphrase", "sentence", {"correct": "no"}),       # grounded, wrong
                    "ng-2": ("keyword", "bare", {"correct": "no", "grounded": "no"})},
                   prompt="p2", tag="refusal2")
        for run in (base, new):
            for qid, row in run["per_question"].items():
                row["gold_context"] = nogold if qid.startswith("ng") else gold
        return base, new

    base, new = runs("partial")                      # mh-1 drops yes -> partial
    out = C.compare(base, new)
    assert out["criteria"]["multi_hop_correct_yes"]["value"].startswith("0/2")
    assert not out["criteria"]["multi_hop_correct_yes"]["pass"]
    assert out["no_gold_outcomes"]["base"]["refused"] == ["ng-1", "ng-2"]
    assert out["no_gold_outcomes"]["new"]["grounded_wrong"] == ["ng-1"]
    assert out["no_gold_outcomes"]["new"]["ungrounded"] == ["ng-2"]

    base, new = runs("yes")
    assert C.compare(base, new)["criteria"]["multi_hop_correct_yes"]["pass"]


def test_the_kept_2b_prompt_is_the_one_dev_refusal_was_scored_with():
    from benchmark.v3.generation import answer_features as F

    results = os.path.join(os.path.dirname(J.__file__), "results", "dev_refusal.json")
    if not os.path.exists(results):
        pytest.skip("no 2b results in this checkout")
    with open(results, encoding="utf-8") as handle:
        assert F.prompt_hash(F.REFUSAL_V1_QA_TEMPLATE) == json.load(handle)["answer_prompt_sha256"]


def test_the_current_prompt_is_the_refusal2_revision():
    from benchmark.v3.generation import answer_features as F
    from src import rag_chain

    assert rag_chain._QA_TEMPLATE not in (F.BASELINE_QA_TEMPLATE, F.REFUSAL_V1_QA_TEMPLATE)
    assert "answer every part that any passage in the context covers" in rag_chain._QA_TEMPLATE
    assert "say a part is not covered only if no passage covers it" in rag_chain._QA_TEMPLATE


def test_the_summary_counts_caveats_with_every_part_in_context():
    from benchmark.v3.generation import answer_features as F

    def f(missing, ctx, refusal=False, answerable=True):
        return {"words": 10, "refusal_exact": refusal, "says_missing": missing,
                "parts_in_answer": "1/2", "parts_in_context": ctx, "answerable": answerable}
    out = {"a": {"current": f(True, "2/2")}, "b": {"current": f(True, "1/2")},
           "c": {"current": f(True, "1/1", refusal=True)}, "d": {"current": None},
           "u": {"current": f(True, "-", answerable=False)}}
    s = F.summarize(out, "current")
    assert s["answers"] == 3 and s["not_cached"] == 1 and s["exact_refusals"] == 1
    assert s["says_missing"] == 2 and s["says_missing_all_parts_in_context"] == ["a"]
