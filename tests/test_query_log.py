"""Per-query logging, Gemini failure classification, and call counting."""

import io

import pytest

from src.query_log import CallCounter, format_query_log, is_quota_error, log_query, redact


class FakeAPIError(Exception):
    def __init__(self, code=None, status=None):
        super().__init__("provider said no")
        self.code = code
        self.status = status


def test_quota_refusals_are_recognised_by_code_or_status():
    assert is_quota_error(FakeAPIError(code=429))
    assert is_quota_error(FakeAPIError(status="RESOURCE_EXHAUSTED"))
    assert not is_quota_error(FakeAPIError(code=500, status="INTERNAL"))
    assert not is_quota_error(ValueError("the text 429 alone is not a quota error"))


def test_the_real_google_genai_quota_error_is_recognised():
    errors = pytest.importorskip("google.genai.errors")
    exc = errors.ClientError(
        429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}}
    )
    assert is_quota_error(exc)


def test_log_line_has_the_fields_and_only_the_error_type_and_code():
    line = format_query_log("quota", "shared", "gemini-3.5-flash-lite", 2.345, 3,
                            error=FakeAPIError(code=429))
    assert line == (
        "query status=quota key=shared model=gemini-3.5-flash-lite "
        "seconds=2.3 gemini_calls=3 error=FakeAPIError code=429"
    )
    assert "provider said no" not in line


def test_log_query_writes_exactly_one_line():
    stream = io.StringIO()
    log_query("ok", "own", "gemini-3.5-flash", 1.0, 2, stream=stream)
    assert stream.getvalue() == "query status=ok key=own model=gemini-3.5-flash seconds=1.0 gemini_calls=2\n"


def test_redact_blanks_the_secret_and_ignores_an_empty_one():
    assert redact("key AIza-123 rejected", "AIza-123") == "key [redacted] rejected"
    assert redact("nothing to hide", "") == "nothing to hide"
    assert redact("nothing to hide", None) == "nothing to hide"


# ── Call counting through the real RAG chain, with a fake Gemini client ──────

class _Response:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    def __init__(self, api_key=None):
        self.models = self

    def generate_content(self, model, contents):
        return _Response("Hi there!")


@pytest.fixture
def rag_chain(monkeypatch):
    genai = pytest.importorskip("google.genai")
    monkeypatch.setattr(genai, "Client", _FakeClient)
    from src import rag_chain as module
    return module


def test_every_gemini_call_is_reported_to_the_counter(rag_chain):
    counter = CallCounter()
    chain = rag_chain.build_rag_chain("fake-key", "gemini-3.5-flash-lite", None)

    result = chain.invoke({"input": "hello", "chat_history": [], "on_gemini_call": counter})

    assert result["answer"] == "Hi there!"
    assert counter.count == 1


def test_counting_is_optional(rag_chain):
    chain = rag_chain.build_rag_chain("fake-key", "gemini-3.5-flash-lite", None)
    assert chain.invoke({"input": "hello", "chat_history": []})["answer"] == "Hi there!"
