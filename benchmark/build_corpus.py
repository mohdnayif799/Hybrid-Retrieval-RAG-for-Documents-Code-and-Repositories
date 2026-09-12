"""
Reproducible benchmark corpus: a SHA-pinned GitHub repository plus the
project's own documents, ingested through the real pipeline.

Why a third store prefix
------------------------
``store_paths`` defines two namespaces and neither is usable here:

  * ``chroma_db_*``   belongs to app.py's per-chat stores. A benchmark store
    living there would show up as a phantom chat.
  * ``chroma_eval_*`` is deleted wholesale by ``stale_eval_store_dirs()`` on
    the next evaluation run, which is exactly what a benchmark must not be.

So the benchmark owns ``chroma_bench_*``. Neither existing glob matches it,
and this module deliberately does not add the constant to ``store_paths`` —
that module is source, and this is evaluation scaffolding.

Nothing here calls Gemini. Retrieval runs entirely before generation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import uuid

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CACHE_DIR = os.path.join(BENCHMARK_DIR, ".cache")
CORPUS_DIR = os.path.join(BENCHMARK_DIR, "corpus")
MANIFEST_PATH = os.path.join(BENCHMARK_DIR, "corpus_manifest.json")

# Benchmark store namespace. Matched by neither store_paths glob.
BENCH_STORE_PREFIX = "chroma_bench"

# ── The pin ──────────────────────────────────────────────────────────────────
# A commit SHA, never a branch: a branch makes the benchmark irreproducible.
REPO_OWNER = "psf"
REPO_NAME = "requests"
REPO_SHA = "dae7ef63b4df6eded86637f251fc4e3a06c3b479"

DOCUMENTS = (
    "Deep_Learning_Unit1_Quick_Notes.pdf",
    "space_planets_moons_advanced.txt",
    "V_SEM_3_to_5_units_ETCE.docx",
)

# Recorded so a corrupted or substituted corpus fails loudly instead of
# silently changing the numbers.
EXPECTED_SHA256 = {
    "Deep_Learning_Unit1_Quick_Notes.pdf":
        "974bc4248d10cbd678783fdc5d27104992982db5bce9b356ee4589f1d19ba278",
    "space_planets_moons_advanced.txt":
        "581fd18b9738b3efc7ee58b376dd73ae7d59ea4f99db85f3fb25a76253ace4af",
    # Added in v2. A .docx has no page metadata - Docx2txtLoader returns one
    # Document at page 0 - so all 99 of its chunks share a single (file, page)
    # locator. Ground truth for this document is therefore start_index RANGES,
    # not file/page. See benchmark/metrics.chunk_matches.
    "V_SEM_3_to_5_units_ETCE.docx":
        "ad6579aab7f10f540749ce59730a1bd34564610af5e14c61510057fc7eb67423",
    "_archive": "de310df9b97e0c2b28792e6c51db49d65a54b542899687aa4b7c868653b91fbf",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def new_bench_store_dir(root: str | None = None) -> str:
    """Mirrors store_paths' naming scheme, in the benchmark's own namespace."""
    suffix = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    return os.path.join(root or PROJECT_ROOT, f"{BENCH_STORE_PREFIX}_{suffix}")


# ── Repository half ──────────────────────────────────────────────────────────

def archive_path() -> str:
    return os.path.join(CACHE_DIR, f"{REPO_OWNER}-{REPO_NAME}-{REPO_SHA}.tar.gz")


def fetch_archive(token: str | None = None) -> bytes:
    """Return the pinned tarball, downloading it only once."""
    from src.repo_ingestion import download_archive, parse_repo_url, resolve_repository

    path = archive_path()
    if os.path.exists(path):
        with open(path, "rb") as handle:
            return handle.read()

    os.makedirs(CACHE_DIR, exist_ok=True)
    repo = parse_repo_url(
        f"https://github.com/{REPO_OWNER}/{REPO_NAME}/tree/{REPO_SHA}"
    )
    ref, _size_kb = resolve_repository(repo, token=token)
    if ref != REPO_SHA:
        raise RuntimeError(f"resolved ref {ref!r} is not the pinned SHA {REPO_SHA!r}")
    raw = download_archive(repo, ref, token=token).getvalue()
    with open(path, "wb") as handle:
        handle.write(raw)
    return raw


def load_repo_documents(token: str | None = None):
    """Pinned repository -> Documents, through the real ingestion path."""
    from src.repo_ingestion import (
        DEFAULT_LIMITS, RepoRef, read_archive, repo_files_to_documents,
    )

    raw = fetch_archive(token=token)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256["_archive"]:
        raise RuntimeError(
            f"archive sha256 {digest} != expected {EXPECTED_SHA256['_archive']}"
        )

    files, report = read_archive(io.BytesIO(raw), limits=DEFAULT_LIMITS)
    repo = RepoRef(owner=REPO_OWNER, name=REPO_NAME, ref=REPO_SHA)
    return repo_files_to_documents(files, repo, REPO_SHA), files, report


# ── Document half ────────────────────────────────────────────────────────────

def document_paths() -> list[str]:
    paths = []
    for name in DOCUMENTS:
        path = os.path.join(CORPUS_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"benchmark corpus document missing: {path}")
        digest = sha256_file(path)
        if digest != EXPECTED_SHA256[name]:
            raise RuntimeError(f"{name} sha256 {digest} != expected {EXPECTED_SHA256[name]}")
        paths.append(path)
    return paths


def load_doc_documents():
    """Project documents -> Documents, through the real loader."""
    from src.data_ingestion import load_documents
    return load_documents(document_paths())


# ── Combined corpus ──────────────────────────────────────────────────────────

def build_chunks(token: str | None = None):
    """Every chunk in the benchmark corpus, via the single chunk_documents entry."""
    from src.chunking import chunk_documents

    repo_docs, files, report = load_repo_documents(token=token)
    doc_docs = load_doc_documents()
    chunks = chunk_documents(repo_docs + doc_docs)
    return chunks, {"repo_files": files, "repo_report": report,
                    "n_repo_docs": len(repo_docs), "n_doc_docs": len(doc_docs)}


def build_store(chroma_dir: str | None = None, token: str | None = None,
                progress: bool = True):
    """
    Embed the corpus into a fresh chroma_bench_* store with real MiniLM.

    Uses vector_store._embed_in_batches, which is what both public creators
    (create_chat_vector_store / build_vector_store) call. Going through it
    keeps the one-client-per-directory discipline those functions document;
    the only thing not reused is their directory naming, which would put the
    store in the wrong namespace.
    """
    from src import vector_store

    chroma_dir = chroma_dir or new_bench_store_dir()

    def report(done, total):
        if progress:
            print(f"    embedded {done}/{total}", flush=True)

    chunks, meta = build_chunks(token=token)
    vector_store._embed_in_batches(chunks, chroma_dir, report if progress else None)
    return chroma_dir, chunks, meta


def write_manifest(chroma_dir: str, chunks, meta) -> dict:
    from collections import Counter

    per_source = Counter(c.metadata.get("source", "?") for c in chunks)
    manifest = {
        "repo": {
            "slug": f"{REPO_OWNER}/{REPO_NAME}",
            "sha": REPO_SHA,
            "archive_sha256": EXPECTED_SHA256["_archive"],
            "indexable_files": meta["n_repo_docs"],
            "skipped": dict(meta["repo_report"].skipped),
        },
        "documents": [
            {"file": name, "sha256": EXPECTED_SHA256[name],
             "bytes": os.path.getsize(os.path.join(CORPUS_DIR, name))}
            for name in DOCUMENTS
        ],
        "chunks": {
            "total": len(chunks),
            "repo": sum(1 for c in chunks if c.metadata.get("source_type") == "repo"),
            "upload": sum(1 for c in chunks if c.metadata.get("source_type") != "repo"),
        },
        "store_dir": os.path.basename(chroma_dir),
        "files_with_chunks": len(per_source),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


if __name__ == "__main__":
    started = time.time()
    store_dir, chunks, meta = build_store()
    manifest = write_manifest(store_dir, chunks, meta)
    print(json.dumps(manifest, indent=2))
    print(f"\nstore: {store_dir}")
    print(f"built in {time.time() - started:.1f}s")
