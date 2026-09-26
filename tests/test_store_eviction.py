"""
Evicting a chat store really frees it: src/vector_store.evict_store.

Uses the real ChromaDB from constraints.txt, with fake embeddings so no model
is downloaded. evict_store() reaches into chromadb internals to stop the
per-directory "system" (see _release_chroma_system); these tests are what
keep that honest across chromadb upgrades.
"""

import os
import sys

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("langchain_community")

from chromadb.api.shared_system_client import SharedSystemClient  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from src import hybrid_retrieval, vector_store  # noqa: E402
from src.store_registry import StoreBudget, StoreRegistry  # noqa: E402

DOCS = [Document(page_content="alpha beta gamma", metadata={"source": "a.txt"})]


@pytest.fixture
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(vector_store, "get_embeddings", lambda: DeterministicFakeEmbedding(size=8))


def _open_files_under(path):
    """Paths under ``path`` this process still holds open (Linux only)."""
    if not sys.platform.startswith("linux"):
        return []
    root = os.path.realpath(path)
    held = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith(root):
            held.append(target)
    return held


def test_evict_store_stops_chroma_closes_files_and_deletes_the_directory(tmp_path, fake_embeddings):
    chroma_dir = str(tmp_path / "chroma_db_test")
    vector_store.load_vector_store(chroma_dir).add_documents(DOCS)
    assert chroma_dir in SharedSystemClient._identifier_to_system

    vector_store.evict_store(chroma_dir)

    assert chroma_dir not in SharedSystemClient._identifier_to_system
    assert not os.path.exists(chroma_dir)
    assert _open_files_under(chroma_dir) == []
    assert not vector_store.store_exists(chroma_dir)


def test_creating_a_store_over_budget_evicts_the_oldest(tmp_path, fake_embeddings, monkeypatch):
    monkeypatch.setenv("RAG_STORE_DIR", str(tmp_path))
    registry = StoreRegistry(StoreBudget(max_count=1, max_mb=1000, idle_seconds=1e9),
                             release=vector_store.evict_store)
    monkeypatch.setattr(vector_store, "STORE_REGISTRY", registry)

    _, first = vector_store.create_chat_vector_store(DOCS)
    _, second = vector_store.create_chat_vector_store(DOCS)

    assert os.path.dirname(first) == str(tmp_path)
    assert not os.path.exists(first) and first not in registry
    assert os.path.isdir(second) and second in registry
    vector_store.evict_store(second)


class _RecordingCache:
    """Stands in for the cached load_bm25_index, recording per-entry clears."""

    def __init__(self):
        self.cleared = []

    def clear(self, *args):
        self.cleared.append(args)


def test_bm25_entry_superseded_by_an_append_is_dropped(monkeypatch):
    cache = _RecordingCache()
    monkeypatch.setattr(hybrid_retrieval, "load_bm25_index", cache)
    store = "chroma_db_bm25_bookkeeping"

    hybrid_retrieval._record_bm25_count(store, 5)
    hybrid_retrieval._record_bm25_count(store, 5)     # same count: nothing stale
    hybrid_retrieval._record_bm25_count(store, 7)     # after an append
    assert cache.cleared == [(store, 5)]

    hybrid_retrieval.forget_bm25_index(store)         # eviction
    assert cache.cleared == [(store, 5), (store, 7)]
    hybrid_retrieval.forget_bm25_index(store)         # already forgotten: no-op
    assert len(cache.cleared) == 2
