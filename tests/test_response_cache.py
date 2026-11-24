"""Rules for when a chat may reuse a cached answer (src/response_cache.py)."""

import pytest
from langchain_core.documents import Document

from src.response_cache import (
    NO_DOCUMENTS,
    batch_digest,
    cache_key,
    depends_on_history,
    document_set_hash,
    is_cacheable,
    normalize_question,
)


def _chunks(*texts, source="notes.txt"):
    return [Document(page_content=t, metadata={"source": source}) for t in texts]


# ── Document set ──────────────────────────────────────────────────────────────

def test_a_chat_without_documents_has_a_fixed_document_set():
    assert document_set_hash([]) == NO_DOCUMENTS


def test_adding_documents_changes_the_document_set():
    first = batch_digest(_chunks("alpha", "beta"))
    second = batch_digest(_chunks("gamma", source="more.pdf"))
    assert document_set_hash([first]) != document_set_hash([first, second])
    assert document_set_hash([first]) != NO_DOCUMENTS


def test_document_set_ignores_the_order_batches_were_added_in():
    a, b = batch_digest(_chunks("alpha")), batch_digest(_chunks("beta"))
    assert document_set_hash([a, b]) == document_set_hash([b, a])


def test_batch_digest_depends_on_text_and_source():
    base = batch_digest(_chunks("alpha"))
    assert base == batch_digest(_chunks("alpha"))
    assert base != batch_digest(_chunks("alpha!"))
    assert base != batch_digest(_chunks("alpha", source="other.txt"))
    # Chunk boundaries matter too: "ab" + "c" is not "a" + "bc".
    assert batch_digest(_chunks("ab", "c")) != batch_digest(_chunks("a", "bc"))


# ── Key ───────────────────────────────────────────────────────────────────────

def test_the_same_question_in_two_chats_has_two_keys():
    docs = document_set_hash([])
    assert cache_key("chat-a", docs, "What is X?") != cache_key("chat-b", docs, "What is X?")


def test_the_same_question_after_documents_were_added_has_a_new_key():
    before = document_set_hash([])
    after = document_set_hash([batch_digest(_chunks("alpha"))])
    assert cache_key("c", before, "What is X?") != cache_key("c", after, "What is X?")


def test_case_spacing_and_trailing_punctuation_do_not_change_the_key():
    docs = document_set_hash([])
    assert (cache_key("c", docs, "  What is   X? ")
            == cache_key("c", docs, "what is x")
            == cache_key("c", docs, "What is X!"))
    assert normalize_question("What is X?") == "what is x"


# ── Follow-ups ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("question", [
    "and the next one?",
    "What about its timeout?",
    "Why?",
    "Tell me more",
    "How does that work?",
    "What do they return?",
    "Explain the second step",
    "Can you elaborate on the previous answer?",
    "But what does the other function do?",
    "Elaborate",
    "more examples please",
])
def test_follow_ups_that_lean_on_earlier_turns_are_detected(question):
    assert depends_on_history(question)


@pytest.mark.parametrize("question", [
    "What is in the documents?",
    "How does itsdangerous reject a signed token older than max_age?",
    "What is the default timeout in requests.get?",
    "List the planets with rings",
    "Which exception is raised for too many redirects?",
])
def test_standalone_questions_are_not_flagged(question):
    assert not depends_on_history(question)


def test_first_messages_are_always_cacheable():
    # With no history there is nothing for "it" or "next" to refer to.
    assert is_cacheable("and the next one?", has_history=False)
    assert is_cacheable("What is in the documents?", has_history=False)


def test_later_messages_are_cacheable_only_if_they_stand_alone():
    assert is_cacheable("What is in the documents?", has_history=True)
    assert not is_cacheable("and the next one?", has_history=True)
