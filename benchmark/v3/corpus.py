"""
The v3 benchmark corpus: the same pinned repository and documents as v2,
loaded through the real ingestion and chunking code.

Differences from ``benchmark.build_corpus``:

* The repository archive is verified by a CONTENT digest (the sorted paths and
  text hashes of every indexable file) rather than by the tarball's bytes. Two
  archives of the same commit can differ in gzip framing and timestamps while
  holding identical files, so the byte hash rejected a valid ``git archive`` of
  the pinned commit. The content digest is what the chunks actually depend on.
* ``archive`` can point at a local tarball, so the benchmark runs where
  codeload.github.com is unreachable. Without it, ``build_corpus.fetch_archive``
  is used as before (download once, cache under benchmark/.cache).
* Parent texts (whole files and pages) are returned alongside the chunks, so
  gold spans can be resolved to character ranges inside them.
"""

from __future__ import annotations

import hashlib
import io
import ntpath
import os
import sys

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark import build_corpus

# sha256 over "path\0sha256(text)\n" for every indexable file, sorted by path,
# for psf/requests at build_corpus.REPO_SHA under DEFAULT_LIMITS.
EXPECTED_CONTENT_DIGEST = "7a616ffd01feb36d9c4b46a76db3b9a0c360b7fdd0f17d99201e82b5562702c2"
EXPECTED_CHUNKS = 1222


def content_digest(files) -> str:
    """Digest of the indexable files an archive yields, independent of gzip bytes."""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda f: f.path):
        h.update(f.path.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(f.text.encode("utf-8")).hexdigest().encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def load_repo_files(archive: str | None = None):
    """Indexable files of the pinned repository, content-verified."""
    from src.repo_ingestion import DEFAULT_LIMITS, read_archive

    if archive:
        with open(archive, "rb") as handle:
            raw = handle.read()
    else:
        raw = build_corpus.fetch_archive()
    files, report = read_archive(io.BytesIO(raw), limits=DEFAULT_LIMITS)
    digest = content_digest(files)
    if digest != EXPECTED_CONTENT_DIGEST:
        raise RuntimeError(
            f"repository content digest {digest} != expected {EXPECTED_CONTENT_DIGEST}; "
            "the archive is not psf/requests at the pinned commit, or DEFAULT_LIMITS "
            "(RAG_REPO_* environment variables) changed which files are indexed"
        )
    return files, report


def doc_file_key(doc) -> str:
    """File identity used by gold spans: repo-relative path, or document basename."""
    source = doc.metadata.get("source", "")
    if doc.metadata.get("source_type") == "repo":
        return source
    return ntpath.basename(source)


def load_corpus(archive: str | None = None):
    """
    Returns (chunks, parents, meta).

    ``parents`` maps (file, page) -> full parent text, where file is the
    repository-relative path or the document basename and page is the loader's
    page metadata (0 for repository files, .txt and .docx; 0-based for PDFs).
    """
    from src.chunking import chunk_documents
    from src.repo_ingestion import RepoRef, repo_files_to_documents

    files, report = load_repo_files(archive)
    repo = RepoRef(owner=build_corpus.REPO_OWNER, name=build_corpus.REPO_NAME,
                   ref=build_corpus.REPO_SHA)
    repo_docs = repo_files_to_documents(files, repo, build_corpus.REPO_SHA)
    doc_docs = build_corpus.load_doc_documents()   # sha256-verified documents

    parents: dict[tuple, str] = {}
    for doc in repo_docs + doc_docs:
        key = (doc_file_key(doc), doc.metadata.get("page", 0))
        if key in parents:
            raise RuntimeError(f"duplicate parent document {key}")
        parents[key] = doc.page_content

    chunks = chunk_documents(repo_docs + doc_docs)
    if len(chunks) != EXPECTED_CHUNKS:
        raise RuntimeError(f"expected {EXPECTED_CHUNKS} chunks, got {len(chunks)}")
    meta = {"repo_files": len(files), "skipped": dict(report.skipped),
            "content_digest": EXPECTED_CONTENT_DIGEST, "n_chunks": len(chunks)}
    return chunks, parents, meta
