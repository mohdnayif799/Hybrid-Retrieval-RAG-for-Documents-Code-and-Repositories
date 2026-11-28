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
    def __init__(self, text):
        self.page_content = text


def _fake_build_chain(api_key, model, chroma_dir, llm=None):
    class Chain:
        def invoke(self, inputs):
            reply = llm(f"PROMPT for: {inputs['input']}")
            return {"answer": reply.text, "context": [_Doc(SECRET_CHUNK)]}
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
