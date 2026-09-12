#!/usr/bin/env python3
"""
Retrieval-only smoke test over a REAL repository.

Packs this project's own source tree into a GitHub-shaped tarball and pushes it
through the entire ingestion path — archive reading, filtering, language
chunking, line numbering, BM25 indexing, RRF fusion, citation rendering.

Real: the source files, the archive, the filters, the chunker, BM25, RRF,
      metadata and citations.
Not real: the dense retriever (stubbed) and Chroma, which need torch and
      chromadb. Zero Gemini API calls — retrieval runs entirely before
      generation, so none are needed.

    python smoke_repo_retrieval.py
"""

import io
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.chunking import chunk_documents                        # noqa: E402
from src.citations import build_inline_citations, format_reference  # noqa: E402
from src.hybrid_retrieval import BM25Index, HybridRetriever, chunk_key  # noqa: E402
from src.repo_ingestion import (                                # noqa: E402
    RepoRef, parse_repo_url, read_archive, repo_files_to_documents,
)

ROOT = "uzair-rag-doc-qa-abc1234"
PROJECT = os.path.dirname(os.path.abspath(__file__))

checks_passed = 0
checks_failed = 0


def check(label, condition, detail=""):
    global checks_passed, checks_failed
    if condition:
        checks_passed += 1
        print(f"  [PASS] {label}" + (f" — {detail}" if detail else ""))
    else:
        checks_failed += 1
        print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))


def build_tarball_from_project():
    """Pack the real project tree exactly as GitHub would serve it."""
    buf = io.BytesIO()
    added = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(PROJECT):
            # Mirror what GitHub actually serves. A repository tarball never
            # contains venv/, node_modules/ or build output — those are
            # gitignored. Packing them would make this a test of the local
            # folder rather than of a repository, and a virtualenv holding
            # torch alone is several GB.
            dirnames[:] = [d for d in dirnames
                           if d not in {".git", "__pycache__", ".pytest_cache",
                                        "venv", ".venv", "env", "node_modules",
                                        "site-packages", ".mypy_cache", ".ruff_cache",
                                        ".idea", ".vscode", "dist", "build",
                                        ".pytest-tmp", ".eggs"}
                           and not d.startswith("chroma_")]
            for name in filenames:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, PROJECT).replace(os.sep, "/")
                try:
                    data = open(full, "rb").read()
                except OSError:
                    continue
                info = tarfile.TarInfo(name=f"{ROOT}/{rel}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                added += 1
    buf.seek(0)
    return buf, added


def main():
    print("=" * 72)
    print("  REPOSITORY INGESTION SMOKE TEST — real source, real retrieval")
    print("=" * 72)

    print("\n[1/6] URL parsing")
    ref = parse_repo_url("https://github.com/uzair/rag-doc-qa/tree/main")
    check("parses owner/name/ref", (ref.owner, ref.name, ref.ref) == ("uzair", "rag-doc-qa", "main"))

    print("\n[2/6] Archive read + filtering (real project tree)")
    tarball, packed = build_tarball_from_project()
    files, report = read_archive(tarball)
    print(f"       packed {packed} file(s) -> kept {report.kept}, skipped {report.skipped_total}")
    for reason, count in sorted(report.skipped.items(), key=lambda kv: -kv[1]):
        print(f"         - {count} x {reason}")
    check("kept the source files", report.kept >= 8, f"{report.kept} files")
    check("no .git or __pycache__ leaked through",
          not any("/.git/" in f.path or "__pycache__" in f.path for f in files))
    check("all paths are relative", all(not f.path.startswith("/") for f in files))
    check("deterministic ordering", [f.path for f in files] == sorted(f.path for f in files))

    print("\n[3/6] Documents + chunking")
    docs = repo_files_to_documents(files, RepoRef("uzair", "rag-doc-qa"), "main")
    chunks = chunk_documents(docs)
    py = [c for c in chunks if c.metadata["source"].endswith(".py")]
    check("produced chunks", len(chunks) > 20, f"{len(chunks)} chunks from {len(docs)} files")
    check("python files were chunked", len(py) > 10, f"{len(py)} python chunks")
    check("every chunk has a start_line", all("start_line" in c.metadata for c in chunks))
    check("chunk keys unique", len({chunk_key(c) for c in chunks}) == len(chunks))
    check("metadata is Chroma-scalar",
          all(isinstance(v, (str, int, float, bool)) or v is None
              for c in chunks for v in c.metadata.values()))

    print("\n[4/6] Line-number accuracy (verified against the real files)")
    by_source = {}
    for c in chunks:
        by_source.setdefault(c.metadata["source"], []).append(c)
    verified = mismatched = 0
    for f in files:
        lines = f.text.splitlines()
        for c in by_source.get(f.path, []):
            n = c.metadata["start_line"]
            first = c.page_content.splitlines()[0].strip() if c.page_content.splitlines() else ""
            if not first:
                continue
            actual = lines[n - 1].strip() if 0 < n <= len(lines) else ""
            if actual.startswith(first[:25]) or first.startswith(actual[:25]):
                verified += 1
            else:
                mismatched += 1
    check("reported line numbers match file contents",
          mismatched == 0, f"{verified} verified, {mismatched} mismatched")

    print("\n[5/6] Hybrid retrieval (real BM25 + real RRF)")
    index = BM25Index(chunks)
    print(f"       BM25 indexed {len(index)} chunk(s), skipped {index.skipped_empty} empty")
    check("BM25 index built", index.ready)

    queries = [
        ("build hybrid retriever", "build_hybrid_retriever"),
        ("reciprocal rank fusion", "reciprocal_rank_fusion"),
        ("stale eval store dirs", "stale_eval_store_dirs"),
        ("safe member path", "safe_member_path"),
        ("start line of", "start_line_of"),
    ]
    for query, expected in queries:
        hits = index.search(query, k=5)
        found = any(expected in d.page_content for d, _ in hits)
        check(f"lexical: {query!r} finds {expected}", found,
              f"top hit {hits[0][0].metadata['source']}" if hits else "no hits")

    print("\n[6/6] Fusion, provenance and citations")
    target = "reciprocal_rank_fusion"
    dense_only = [c for c in chunks if target not in c.page_content][:10]
    retriever = HybridRetriever(
        vector_retriever=type("Stub", (), {"invoke": lambda self, q: dense_only})(),
        bm25_index=index,
        final_k=5,
    )
    results = retriever.invoke("reciprocal rank fusion")
    check("BM25 rescued a chunk the dense side lacked",
          any(target in r.page_content for r in results))
    check("all results carry provenance", all("retrieval" in r.metadata for r in results))
    check("all results carry an rrf_score", all("rrf_score" in r.metadata for r in results))

    a = [r.page_content for r in retriever.invoke("reciprocal rank fusion")]
    b = [r.page_content for r in retriever.invoke("reciprocal rank fusion")]
    check("retrieval is deterministic", a == b)

    refs = [format_reference(r, i) for i, r in enumerate(results, 1)]
    check("citations are path:line", all(r.rsplit(":", 1)[-1].isdigit() for r in refs))
    print("\n       Sources block as the user would see it:")
    for line in build_inline_citations(results).split("\n\n"):
        print(f"         {line}")
    print("\n       Top results:")
    for i, r in enumerate(results, 1):
        print(f"         {i}. {format_reference(r, i):<45} [{r.metadata['retrieval']}]")

    print("\n" + "=" * 72)
    print(f"  {checks_passed} passed, {checks_failed} failed")
    print("=" * 72)
    return 1 if checks_failed else 0


if __name__ == "__main__":
    sys.exit(main())
