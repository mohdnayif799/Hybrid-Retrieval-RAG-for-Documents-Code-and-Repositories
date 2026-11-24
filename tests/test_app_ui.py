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


# ── Response cache ────────────────────────────────────────────────────────────
# The chain is replaced by a counting fake, so these tests see exactly when an
# answer is generated and when it is served from the cache. Each generated
# answer is distinct ("answer 1", "answer 2", ...).

CACHED_CAPTION = "retrieved from session cache"


def _counting_chain(monkeypatch) -> list:
    """Replace build_rag_chain; returns the list of questions it answered."""
    from src import rag_chain

    asked: list = []

    class FakeChain:
        def invoke(self, inputs):
            asked.append(inputs["input"])
            return {"answer": f"answer {len(asked)}", "context": []}

    monkeypatch.setattr(rag_chain, "build_rag_chain",
                        lambda api_key, model, chroma_dir: FakeChain())
    return asked


def _ask(at: AppTest, question: str) -> AppTest:
    return at.chat_input[0].set_value(question).run()


def _served_from_cache(at: AppTest) -> bool:
    return any(CACHED_CAPTION in c for c in _captions(at))


def _fake_store(monkeypatch, tmp_path):
    """Indexing succeeds without embedding anything (no model download)."""
    from src import vector_store

    store = str(tmp_path / "chroma_db_fake")
    monkeypatch.setattr(vector_store, "create_chat_vector_store",
                        lambda chunks, progress_callback=None: (None, store))
    monkeypatch.setattr(vector_store, "store_exists", lambda path: True)
    monkeypatch.setattr(vector_store, "get_store_stats",
                        lambda path: {"total_chunks": 1, "per_doc": {}})


def test_a_repeated_question_is_served_from_the_cache(monkeypatch):
    _fake_gemini(monkeypatch, reply="A title")
    asked = _counting_chain(monkeypatch)
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    _ask(at, QUESTION)
    _ask(at, QUESTION)

    assert not at.exception
    assert len(asked) == 1
    assert _served_from_cache(at)


def test_a_document_added_after_a_cached_answer_invalidates_it(monkeypatch, tmp_path):
    _fake_gemini(monkeypatch, reply="A title")
    asked = _counting_chain(monkeypatch)
    _fake_store(monkeypatch, tmp_path)
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    _ask(at, QUESTION)                                   # cached, no documents
    at.run()                                             # idle rerun shows the chip
    # Index the bundled example repository through the real chip: the real
    # chunking and _index_chunks run; only the embedding store is faked.
    at.button(key="example_repo").click().run()
    chat = at.session_state.chats[at.session_state.active_id]
    assert chat["chroma_dir"] and len(chat["doc_digests"]) == 1

    _ask(at, QUESTION)

    assert not at.exception
    assert asked == [QUESTION, QUESTION]                 # generated again
    assert not _served_from_cache(at)
    assert [m["content"] for m in chat["messages"] if m["role"] == "assistant"] == [
        "answer 1", "answer 2"]


def test_the_same_question_in_two_chats_is_answered_in_each(monkeypatch):
    _fake_gemini(monkeypatch, reply="A title")
    asked = _counting_chain(monkeypatch)
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    _ask(at, QUESTION)
    first_chat = at.session_state.active_id
    at.button(key="new_chat").click().run()
    assert at.session_state.active_id != first_chat
    _ask(at, QUESTION)

    assert not at.exception
    assert len(asked) == 2
    assert not _served_from_cache(at)


def test_a_follow_up_that_depends_on_history_bypasses_the_cache(monkeypatch):
    _fake_gemini(monkeypatch, reply="A title")
    asked = _counting_chain(monkeypatch)
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    _ask(at, QUESTION)
    _ask(at, "and the next one?")
    _ask(at, "and the next one?")

    assert not at.exception
    assert len(asked) == 3
    assert not _served_from_cache(at)
    chat = at.session_state.chats[at.session_state.active_id]
    assert len(chat["response_cache"]) == 1              # only the first question


# ── Sources toggle ────────────────────────────────────────────────────────────
# Only the latest answer shows its sources in full; each earlier answer keeps
# its own sources in a closed "Sources" expander directly beneath it.

SOURCES = "Sources"
CHUNKS = "\U0001f4da Source Chunks Used"


def _chain_with_sources(monkeypatch, sources_by_question: dict) -> None:
    """Each question is answered with chunks from its own file (or none)."""
    from langchain_core.documents import Document

    from src import rag_chain

    class FakeChain:
        def invoke(self, inputs):
            name = sources_by_question[inputs["input"]]
            docs = [] if name is None else [
                Document(page_content=f"Text from {name}.",
                         metadata={"source": name, "source_type": "upload",
                                   "start_index": 0})]
            return {"answer": f"Answer citing {name}.", "context": docs}

    monkeypatch.setattr(rag_chain, "build_rag_chain",
                        lambda api_key, model, chroma_dir: FakeChain())


def _expanders(at: AppTest, label: str) -> list:
    return [e for e in at.expander if e.label == label]


def _text(element) -> str:
    return " ".join(m.value for m in element.markdown)


def _full_citation_blocks(at: AppTest) -> list[str]:
    """The "Sources:" lists drawn in full (outside any expander)."""
    return [m.value for m in at.markdown if m.value.startswith("---\n**Sources:**")]


QUESTIONS = {"What is in alpha?": "alpha.txt",
             "What is in beta?": "beta.txt",
             "What is in gamma?": "gamma.txt"}


def test_a_single_answer_shows_its_sources_in_full(monkeypatch):
    _fake_gemini(monkeypatch, reply="A title")
    _chain_with_sources(monkeypatch, QUESTIONS)
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    _ask(at, "What is in alpha?")

    assert not at.exception
    assert _expanders(at, SOURCES) == []
    blocks = _full_citation_blocks(at)
    assert len(blocks) == 1 and "alpha.txt" in blocks[0]
    assert len(_expanders(at, CHUNKS)) == 1


def test_earlier_answers_keep_their_own_sources_collapsed(monkeypatch):
    _fake_gemini(monkeypatch, reply="A title")
    _chain_with_sources(monkeypatch, QUESTIONS)
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    _ask(at, "What is in alpha?")
    _ask(at, "What is in beta?")

    assert not at.exception
    earlier = _expanders(at, SOURCES)
    assert len(earlier) == 1
    assert earlier[0].proto.expanded is False
    assert "alpha.txt" in _text(earlier[0]) and "beta.txt" not in _text(earlier[0])
    blocks = _full_citation_blocks(at)
    assert len(blocks) == 1 and "beta.txt" in blocks[0] and "alpha.txt" not in blocks[0]
    assert len(_expanders(at, CHUNKS)) == 1          # only the latest answer's

    _ask(at, "What is in gamma?")

    earlier = _expanders(at, SOURCES)
    assert [("alpha.txt" in _text(e), "beta.txt" in _text(e)) for e in earlier] == [
        (True, False), (False, True)]               # each under its own answer, in order
    assert all(e.proto.expanded is False for e in earlier)
    assert "gamma.txt" in _full_citation_blocks(at)[0]


def test_the_latest_answer_stays_in_full_on_an_idle_rerun(monkeypatch):
    _fake_gemini(monkeypatch, reply="A title")
    _chain_with_sources(monkeypatch, QUESTIONS)
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    _ask(at, "What is in alpha?")
    _ask(at, "What is in beta?")
    at.run()                                        # e.g. a sidebar click

    assert len(_expanders(at, SOURCES)) == 1
    blocks = _full_citation_blocks(at)
    assert len(blocks) == 1 and "beta.txt" in blocks[0]


def test_a_latest_answer_without_sources_leaves_earlier_ones_collapsed(monkeypatch):
    _fake_gemini(monkeypatch, reply="A title")
    _chain_with_sources(monkeypatch, {**QUESTIONS, "Tell me a joke please": None})
    at = _app(monkeypatch, FAKE_SHARED_KEY).run()

    _ask(at, "What is in alpha?")
    _ask(at, "Tell me a joke please")
    at.run()

    assert not at.exception
    assert len(_expanders(at, SOURCES)) == 1
    assert _full_citation_blocks(at) == []
