import os
import shutil
import sys
import time
import gc
import streamlit as st
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.store_paths import (
    new_chat_store_dir,
    new_eval_store_dir,
    stale_eval_store_dirs,
)
from src.store_registry import StoreRegistry, budget_from_env

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunks are embedded in batches so the UI can report progress and so a large
# repository does not sit inside one multi-minute call. 256 is a heuristic:
# large enough that per-batch overhead stays negligible, small enough that the
# progress bar moves visibly. Not experimentally tuned.
EMBED_BATCH_SIZE = 256


@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once and cache it for the session."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


# ── Used by evaluate.py ───────────────────────────────────────────────────────

def build_vector_store(chunks: list, progress_callback=None) -> tuple:
    """
    Create a new EVALUATION store and clean up previous evaluation stores.
    Used only by evaluate.py.

    Evaluation stores live under their own ``chroma_eval_`` prefix so the
    cleanup below cannot reach the per-chat stores that app.py creates. See
    src/store_paths.py for why this is a namespace rather than a guard clause.

    Returns (store, chroma_dir) — signature unchanged so evaluate.py keeps
    working without modification.
    """
    chroma_dir = new_eval_store_dir()
    store = _embed_in_batches(chunks, chroma_dir, progress_callback)
    print(f"[INFO] Evaluation store built at '{chroma_dir}'.")
    _cleanup_old_eval_dirs(keep=chroma_dir)
    return store, chroma_dir


def _cleanup_old_eval_dirs(keep: str):
    """Delete evaluation stores from previous runs. Never touches chat stores."""
    for dir_path in stale_eval_store_dirs(keep):
        try:
            shutil.rmtree(dir_path)
            print(f"[INFO] Deleted old evaluation store: '{dir_path}'.")
        except Exception:
            pass


# ── Chat-store lifecycle ─────────────────────────────────────────────────────
# Every chat store this process creates is registered here and evicted when
# idle or over budget (policy and limits: src/store_registry.py).

def _release_chroma_system(chroma_dir: str) -> None:
    """
    Stop Chroma's shared "system" for one directory, closing its files.

    chromadb keeps one system per persist directory in a class-level registry,
    refcounted by the clients that use it, and a client that is merely dropped
    (as the cached one is when evicted, or the temporary one in
    append_to_vector_store) never decrements that count. Deleting the
    directory without this unlinks files that are still open, which on Cloud
    Run's in-memory filesystem frees nothing. This reaches into chromadb
    internals, so tests/test_store_eviction.py pins it against the chromadb
    version in constraints.txt.
    """
    from chromadb.api.shared_system_client import SharedSystemClient as shared

    with shared._refcount_lock:
        shared._identifier_to_refcount.pop(chroma_dir, None)
        system = shared._identifier_to_system.pop(chroma_dir, None)
    if system is not None:
        system.stop()


def evict_store(chroma_dir: str) -> None:
    """Release everything a chat store holds in this process, then delete it."""
    from src.hybrid_retrieval import forget_bm25_index  # lazy: circular import

    forget_bm25_index(chroma_dir)
    load_vector_store.clear(chroma_dir)
    _store_stats_cached.clear()          # keyed by (dir, count); cheap to rebuild
    _release_chroma_system(chroma_dir)
    gc.collect()
    try:
        shutil.rmtree(chroma_dir)
    except FileNotFoundError:
        pass                             # already gone: nothing left to free
    print(f"[INFO] Evicted chat store '{os.path.basename(chroma_dir)}'.",
          file=sys.stderr)


def store_exists(chroma_dir: str) -> bool:
    """False once a chat's store has been evicted (or removed by anything else)."""
    return os.path.isdir(chroma_dir)


STORE_REGISTRY = StoreRegistry(budget_from_env(), release=evict_store)


# ── Used by app.py (per-chat, no cross-chat cleanup) ─────────────────────────

def create_chat_vector_store(chunks: list, progress_callback=None) -> tuple:
    """
    Create a new ChromaDB store for a chat's first ingestion.
    Does NOT delete other chats' stores.

    progress_callback: optional callable(embedded, total) — without it a
    repository of several thousand chunks blocks in one silent call.
    """
    chroma_dir = new_chat_store_dir()
    store = _embed_in_batches(chunks, chroma_dir, progress_callback)
    count = store._collection.count()
    print(f"[INFO] Chat store created at '{chroma_dir}' with {count} chunks.")
    STORE_REGISTRY.register(chroma_dir)
    STORE_REGISTRY.enforce(keep=chroma_dir)
    return store, chroma_dir


def _embed_in_batches(chunks: list, chroma_dir: str, progress_callback=None):
    """
    Embed and persist chunks in batches, reporting progress.

    Uses the cached client from load_vector_store() rather than
    Chroma.from_documents(). That is deliberate: from_documents opens its own
    client, which the caller then holds alongside the cached one that
    retrieval later opens — exactly the two-simultaneous-clients condition
    that append_to_vector_store() documents as the cause of Windows data
    loss. Going through the cache means the create path now holds exactly
    one client for the directory's whole lifetime.
    """
    store = load_vector_store(chroma_dir)   # creates the directory if absent
    total = len(chunks)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        if batch:
            store.add_documents(batch)
        if progress_callback:
            progress_callback(min(start + EMBED_BATCH_SIZE, total), total)
    return store


def append_to_vector_store(chunks: list, chroma_dir: str, progress_callback=None):
    """
    Append new document chunks to an existing ChromaDB store.

    Root cause of multi-doc bug: having TWO Chroma client objects pointing
    at the same SQLite file simultaneously causes locking and data loss on
    Windows. Fix: clear the Streamlit-cached connection BEFORE opening the
    append connection, collect garbage so the old handle is released, then
    add documents, then delete the temporary connection.

    The next call to load_vector_store() creates a fresh client that reads
    all data — both the original and the newly appended chunks.
    """
    # Step 1: Release existing cached connection to avoid SQLite lock conflict
    load_vector_store.clear()
    gc.collect()
    time.sleep(0.4)   # give Windows time to release file handles

    # Step 2: Open a fresh connection and append
    embeddings = get_embeddings()
    store = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

    before = store._collection.count()
    # Batched so a repository append reports progress instead of blocking
    # silently. This path deliberately keeps its OWN client and the
    # cache-clear/gc dance above rather than reusing _embed_in_batches, which
    # goes through the cached client — the Windows locking behaviour this
    # function guards against is not reproducible here, so it is left alone.
    total = len(chunks)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        if batch:
            store.add_documents(batch)
        if progress_callback:
            progress_callback(min(start + EMBED_BATCH_SIZE, total), total)
    after = store._collection.count()

    print(f"[INFO] Appended {len(chunks)} chunks. "
          f"Store: {before} → {after} total chunks at '{chroma_dir}'.")

    # Step 3: Explicitly release the append connection
    del store
    gc.collect()
    # Cache is already cleared from Step 1; next load_vector_store call
    # will open a new connection that sees all {after} chunks.

    # The store just grew, so it may now push the process over budget.
    STORE_REGISTRY.touch(chroma_dir)
    STORE_REGISTRY.enforce(keep=chroma_dir)


def get_store_stats(chroma_dir: str) -> dict:
    """
    Return diagnostic info about a vector store.
    Used by the status indicator to show how many chunks are indexed.

    Reads through the cached client rather than opening its own. This runs on
    every sidebar rerun, and a second live Chroma client on the same SQLite
    file is precisely the condition that caused the multi-document data loss
    documented in append_to_vector_store().
    """
    # Runs on every rerun of the chat being viewed, which makes it the "this
    # store is in use" signal for idle eviction.
    STORE_REGISTRY.touch(chroma_dir)
    try:
        total = count_chunks(chroma_dir)
        return _store_stats_cached(chroma_dir, total)
    except Exception as e:
        return {"total_chunks": 0, "per_doc": {}, "error": str(e)}


@st.cache_data(show_spinner=False)
def _store_stats_cached(chroma_dir: str, total: int) -> dict:
    """
    Per-source chunk counts over the WHOLE store.

    Previously this sampled ``limit=500``, which was fine for a PDF and wrong
    for a repository — the breakdown silently became "the first 500 chunks'
    files". Scanning everything is correct but too slow to repeat on every
    Streamlit rerun, so it is cached on (dir, chunk count); ingesting more
    content changes the count and invalidates the entry, the same fingerprint
    trick the BM25 index uses.
    """
    from collections import Counter

    store = load_vector_store(chroma_dir)
    raw = store._collection.get(include=["metadatas"])
    counts = Counter(
        (m or {}).get("source", "unknown") for m in (raw.get("metadatas") or [])
    )
    return {"total_chunks": total, "per_doc": dict(counts)}


def count_chunks(chroma_dir: str) -> int:
    """
    Number of chunks currently indexed. Cheap (a SQLite COUNT) and used as the
    cache fingerprint for the BM25 index, so appends rebuild it automatically.
    """
    try:
        return load_vector_store(chroma_dir)._collection.count()
    except Exception:
        return 0


def export_chunks(chroma_dir: str) -> list:
    """
    Read every indexed chunk back out of the store as LangChain Documents.

    Used to build the BM25 lexical index over exactly the same chunks the
    vector index holds — one ingestion path, one chunking policy, two views of
    it. Goes through the cached client for the same single-connection reason
    as get_store_stats().
    """
    store = load_vector_store(chroma_dir)
    raw = store._collection.get(include=["documents", "metadatas"])
    texts = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    return [
        Document(page_content=text or "", metadata=dict(meta or {}))
        for text, meta in zip(texts, metas)
    ]


@st.cache_resource
def load_vector_store(chroma_dir: str) -> Chroma:
    """Load a ChromaDB store from disk and cache the connection per directory."""
    embeddings = get_embeddings()
    return Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
