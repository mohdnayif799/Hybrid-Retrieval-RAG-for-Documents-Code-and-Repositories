"""
Headless UI tests for app.py via Streamlit's AppTest.

Scope: the demo-key controls and failure handling (kept consistent with
AgentCrew), the model choice, the empty-state "Try an example" chip, and the
notice shown when a chat's store has been evicted.

Nothing here reaches the network or a real model. GOOGLE_API_KEY is always set
explicitly, to "" or to a fake value, so a developer's real key (from the
environment or a .env file; load_dotenv never overrides an existing variable)
can never be used. Tests that get as far as a Gemini call replace
google.genai.Client with a fake.
"""

import os

import pytest

pytest.importorskip("sentence_transformers")  # app.py imports the full ML stack
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
FAKE_SHARED_KEY = "fake-shared-key-for-tests"
SHARED_COUNT_KEY = "_shared_key_query_count"   # demo_key._COUNT_KEY
TIMEOUT = 180   # first run imports torch, Chroma and LangChain
QUESTION = "What is in the documents?"          # not casual, so it is a real query


def _app(monkeypatch, shared_key: str) -> AppTest:
    monkeypatch.setenv("GOOGLE_API_KEY", shared_key)
    monkeypatch.delenv("RAG_MODEL", raising=False)
    return AppTest.from_file(APP, default_timeout=TIMEOUT)


def _captions(at: AppTest) -> list[str]:
    return [c.value for c in at.caption]


def _fake_gemini(monkeypatch, reply=None, error=None):
    """Replace google.genai.Client: every call returns ``reply`` or raises ``error``."""
    genai = pytest.importorskip("google.genai")

    class Response:
        text = reply

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = self

        def generate_content(self, model, contents):
            if error is not None:
                raise error
            return Response()

    monkeypatch.setattr(genai, "Client", FakeClient)


class QuotaError(Exception):
    """Shaped like google-genai's ClientError for a 429."""
    code = 429
    status = "RESOURCE_EXHAUSTED"


# ── Demo key ──────────────────────────────────────────────────────────────────

def test_renders_without_errors_when_no_key_is_configured(monkeypatch):
    at = _app(monkeypatch, "").run()

    assert not at.exception
    assert not any("Demo queries left" in c for c in _captions(at))


def test_shared_key_shows_the_demo_counter(monkeypatch):
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    assert not at.exception
    assert "Demo queries left: 5 of 5" in _captions(at)


def test_own_key_hides_the_demo_counter(monkeypatch):
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    at.text_input(key="own_gemini_key").set_value("visitor-key").run()

    assert not at.exception
    assert not any("Demo queries left" in c for c in _captions(at))


def test_limit_reached_uses_the_agentcrew_wording(monkeypatch):
    at = _app(monkeypatch, FAKE_SHARED_KEY)
    at.session_state[SHARED_COUNT_KEY] = 5
    at.run()

    at.chat_input[0].set_value(QUESTION).run()

    assert not at.exception
    assert "Demo queries left: 0 of 5" in _captions(at)
    warnings = [w.value for w in at.warning]
    assert any("You've used the 5 free demo queries" in w for w in warnings)
    assert any("under 'Use your own API key'" in w for w in warnings)


# ── Gemini outcomes (fake client) ─────────────────────────────────────────────

def test_successful_answer_uses_one_demo_query_and_logs_without_the_question(monkeypatch, capsys):
    _fake_gemini(monkeypatch, reply="A grounded answer.")
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    at.chat_input[0].set_value(QUESTION).run()

    assert not at.exception
    assert "Demo queries left: 4 of 5" in _captions(at)
    err = capsys.readouterr().err
    # First message: one title call plus the conversational answer.
    assert "query status=ok key=shared model=gemini-3.5-flash-lite" in err
    assert "gemini_calls=2" in err
    assert QUESTION not in err and FAKE_SHARED_KEY not in err


def test_quota_refusal_on_the_shared_key_is_explained_and_not_counted(monkeypatch, capsys):
    _fake_gemini(monkeypatch, error=QuotaError("429 RESOURCE_EXHAUSTED"))
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    at.chat_input[0].set_value(QUESTION).run()

    assert not at.exception
    assert ("The shared demo quota is used up for now. Try again later or add "
            "your own key.") in [w.value for w in at.warning]
    assert "Demo queries left: 5 of 5" in _captions(at)
    assert "query status=quota key=shared" in capsys.readouterr().err


def test_other_shared_key_failures_show_a_generic_message(monkeypatch):
    _fake_gemini(monkeypatch, error=RuntimeError("upstream exploded: internal detail"))
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    at.chat_input[0].set_value(QUESTION).run()

    errors = [e.value for e in at.error]
    assert any("The demo is temporarily unavailable" in e for e in errors)
    assert not any("internal detail" in e for e in errors)
    assert "Demo queries left: 5 of 5" in _captions(at)


def test_own_key_failures_show_the_error_with_the_key_redacted(monkeypatch):
    _fake_gemini(monkeypatch, error=RuntimeError("key visitor-key-123 was rejected"))
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()
    at.text_input(key="own_gemini_key").set_value("visitor-key-123").run()

    at.chat_input[0].set_value(QUESTION).run()

    errors = [e.value for e in at.error]
    assert any("Error generating answer" in e and "[redacted]" in e for e in errors)
    assert not any("visitor-key-123" in e for e in errors)


# ── Model choice ──────────────────────────────────────────────────────────────

def test_model_defaults_to_flash_lite_with_flash_as_the_only_alternative(monkeypatch):
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    model = at.selectbox[0]
    assert model.value == "gemini-3.5-flash-lite"
    assert list(model.options) == ["gemini-3.5-flash-lite", "gemini-3.5-flash"]


def test_rag_model_env_var_sets_the_default(monkeypatch):
    at = _app(monkeypatch, FAKE_SHARED_KEY)
    monkeypatch.setenv("RAG_MODEL", "gemini-3.5-flash")
    at.run()

    model = at.selectbox[0]
    assert model.value == "gemini-3.5-flash"
    assert list(model.options) == ["gemini-3.5-flash", "gemini-3.5-flash-lite"]


# ── Empty state and evicted stores ────────────────────────────────────────────

def test_empty_chat_offers_the_example_repository(monkeypatch):
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    chip = at.button(key="example_repo")
    assert "pallets/itsdangerous" in chip.label
    assert any("Try an example" in c for c in _captions(at))


def test_example_without_any_key_explains_instead_of_indexing(monkeypatch):
    at = _app(monkeypatch, "").run()

    at.button(key="example_repo").click().run()

    assert not at.exception
    assert any("Use your own API key" in w.value for w in at.warning)
    assert at.session_state.chats[at.session_state.active_id]["repos"] == []


def test_chat_whose_store_was_evicted_is_told_and_detached(monkeypatch, tmp_path):
    at = _app(monkeypatch, FAKE_SHARED_KEY)
    at.session_state["chats"] = {"c1": {
        "title": "Earlier chat", "messages": [], "lc_history": [],
        "chroma_dir": str(tmp_path / "chroma_db_evicted"),      # does not exist
        "uploaded_files": ["notes.pdf"], "repos": [], "response_cache": {},
        "example_hint_at": None,
    }}
    at.session_state["chat_order"] = ["c1"]
    at.session_state["active_id"] = "c1"
    at.run()

    assert not at.exception
    assert any("were cleared after a period of inactivity" in i.value for i in at.info)
    chat = at.session_state.chats["c1"]
    assert chat["chroma_dir"] is None and chat["uploaded_files"] == []
