"""
When a chat may reuse an earlier answer instead of calling Gemini again.

The app keeps a per-chat cache of answers. It used to be keyed by the
lowercased question alone, which served stale answers in two ways:

* Documents added after an answer was cached did not change the key, so the
  same question returned the old answer and its old sources.
* Chat history was ignored, so a follow-up such as "and the next one?" got its
  first answer back even after the conversation had moved on.

The key is now (chat, document set, question). The document set is a digest
of every batch of chunks the chat has indexed, so it changes whenever
documents are added. A follow-up whose meaning depends on earlier turns is not
cached at all: it is neither read from nor written to the cache.

Stdlib only, so the rules are unit-testable without Streamlit.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

NO_DOCUMENTS = "no-documents"

# Words and openings that make a question lean on earlier turns. Deliberately
# broad: a false positive only costs one Gemini call (the answer is not
# cached), while a false negative would serve an answer to a different
# question. Applied only when the chat already has history.
_REFERENCE_WORDS = frozenset((
    "it", "its", "itself", "this", "that", "these", "those", "they", "them",
    "their", "theirs", "he", "him", "his", "she", "her", "hers", "one", "ones",
    "former", "latter", "above", "below", "previous", "prior", "earlier",
    "next", "last", "again", "same", "other", "another", "else", "more",
    "also", "instead", "first", "second", "third", "fourth", "fifth", "both",
    "either", "neither", "such",
))

_DEPENDENT_OPENINGS = (
    "and", "but", "or", "so", "then", "also", "what about", "how about",
    "why", "how so", "tell me more", "continue", "go on", "elaborate",
    "expand", "explain further", "in more detail",
)

_SHORT_QUESTION_WORDS = 3
_WORD = re.compile(r"[a-z0-9']+")


def normalize_question(question: str) -> str:
    """Case, surrounding whitespace and trailing punctuation do not change the key."""
    text = re.sub(r"\s+", " ", question.strip().lower())
    return text.rstrip("?!. ").strip()


def batch_digest(chunks: Iterable) -> str:
    """
    Digest of one indexed batch: every chunk's source and text, in order.
    Recorded each time a chat indexes documents or a repository.
    """
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(str(chunk.metadata.get("source", "")).encode("utf-8"))
        h.update(b"\0")
        h.update(chunk.page_content.encode("utf-8"))
        h.update(b"\1")
    return h.hexdigest()


def document_set_hash(batch_digests: Iterable[str]) -> str:
    """
    Digest of a chat's whole document set. Order-independent; any new batch
    changes it. A chat with no documents has a fixed value.
    """
    digests = sorted(batch_digests)
    if not digests:
        return NO_DOCUMENTS
    return hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest()


def depends_on_history(question: str) -> bool:
    """True if the question probably needs earlier turns to be understood."""
    text = normalize_question(question)
    words = _WORD.findall(text)
    if len(words) <= _SHORT_QUESTION_WORDS:
        return True
    if any(text == p or text.startswith(p + " ") for p in _DEPENDENT_OPENINGS):
        return True
    return any(w in _REFERENCE_WORDS for w in words)


def is_cacheable(question: str, has_history: bool) -> bool:
    """
    First messages are always cacheable. Later ones only if they stand on
    their own.
    """
    return not (has_history and depends_on_history(question))


def cache_key(chat_id: str, doc_set: str, question: str) -> str:
    """The cache key for ``question`` in this chat with this document set."""
    return "\x1f".join((chat_id, doc_set, normalize_question(question)))
