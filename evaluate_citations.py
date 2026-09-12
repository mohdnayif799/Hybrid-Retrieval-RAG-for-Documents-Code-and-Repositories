#!/usr/bin/env python3
"""
Citation verification. No relevance judgments needed - the source file IS the
ground truth, so every check here is objective.

Four checks:
  1. repo chunks  - start_line points at the chunk's real first line
  2. PDF chunks   - the chunk's text actually occurs on the cited page
  3. text chunks  - the chunk's text occurs at its recorded offset
  4. grouping     - distinct file paths never collapse into one citation group

    python evaluate_citations.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.chunking import start_line_of
from src.citations import citation_group_key, format_reference
from src.repo_ingestion import SOURCE_TYPE_REPO

REPORT_PATH = os.path.join(PROJECT_ROOT, "benchmark", "citations.json")


def normalise(text: str) -> str:
    """Whitespace-insensitive comparison. Line wrapping differs between a
    chunk and a fresh extraction; the words must not."""
    return " ".join(text.split())


def verify_repo(chunks: list, parents: dict) -> dict:
    """start_line must equal the real line number of start_index."""
    checked = mismatched = 0
    failures = []
    for c in chunks:
        if c.metadata.get("source_type") != SOURCE_TYPE_REPO:
            continue
        src = c.metadata["source"]
        text = parents.get((src, None))
        if text is None:
            failures.append({"source": src, "why": "parent file not found"})
            continue
        checked += 1
        expected = start_line_of(text, c.metadata.get("start_index", 0))
        actual = c.metadata.get("start_line")
        if expected != actual:
            mismatched += 1
            failures.append({"source": src, "start_index": c.metadata["start_index"],
                             "expected_line": expected, "recorded_line": actual})
        else:
            # Stronger: the chunk's first line must match the file at that line.
            file_line = text.splitlines()[actual - 1] if actual - 1 < len(text.splitlines()) else None
            chunk_first = c.page_content.splitlines()[0] if c.page_content.splitlines() else ""
            if file_line is not None and chunk_first and chunk_first not in file_line:
                mismatched += 1
                failures.append({"source": src, "line": actual,
                                 "why": "first line of chunk not found on that file line",
                                 "chunk_first": chunk_first[:60],
                                 "file_line": file_line[:60]})
    return {"checked": checked, "mismatched": mismatched, "failures": failures[:20]}


def verify_pdf(chunks: list, pdf_path: str) -> dict:
    """The chunk's text must occur on the page it cites, re-extracted fresh."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages = [normalise(p.extract_text() or "") for p in reader.pages]
    basename = os.path.basename(pdf_path)

    checked = mismatched = 0
    failures = []
    for c in chunks:
        if os.path.basename(c.metadata.get("source", "")) != basename:
            continue
        checked += 1
        page = c.metadata.get("page")
        if page is None or page >= len(pages):
            mismatched += 1
            failures.append({"page": page, "why": "cited page out of range"})
            continue
        if normalise(c.page_content) not in pages[page]:
            mismatched += 1
            failures.append({"page": page,
                             "why": "chunk text not present on cited page",
                             "excerpt": normalise(c.page_content)[:70]})
    return {"checked": checked, "mismatched": mismatched, "failures": failures[:20],
            "n_pages": len(pages)}


def verify_text(chunks: list, txt_path: str) -> dict:
    """The chunk's text must occur at its recorded character offset."""
    with open(txt_path, encoding="utf-8") as handle:
        body = handle.read()
    basename = os.path.basename(txt_path)

    checked = mismatched = 0
    failures = []
    for c in chunks:
        if os.path.basename(c.metadata.get("source", "")) != basename:
            continue
        checked += 1
        start = c.metadata.get("start_index")
        if start is None or body[start:start + len(c.page_content)] != c.page_content:
            mismatched += 1
            failures.append({"start_index": start,
                             "why": "text not found at recorded offset"})
    return {"checked": checked, "mismatched": mismatched, "failures": failures[:20]}


def verify_docx(chunks: list, docx_path: str) -> dict:
    """
    Every .docx chunk's text must occur at its recorded start_index.

    Meaningfully exercised for the first time in v2: this is a 99-chunk,
    78,000-character document whose ground truth is expressed as start_index
    ranges, so if start_index were wrong the benchmark's document half would
    be silently scoring the wrong regions.
    """
    from src.data_ingestion import load_documents

    body = load_documents([docx_path])[0].page_content
    basename = os.path.basename(docx_path)

    checked = mismatched = 0
    failures = []
    for c in chunks:
        if os.path.basename(c.metadata.get("source", "")) != basename:
            continue
        checked += 1
        start = c.metadata.get("start_index")
        if start is None:
            mismatched += 1
            failures.append({"why": "no start_index"})
            continue
        if body[start:start + len(c.page_content)] != c.page_content:
            mismatched += 1
            failures.append({"start_index": start,
                             "why": "text not found at recorded offset",
                             "excerpt": normalise(c.page_content)[:60]})
    return {"checked": checked, "mismatched": mismatched,
            "failures": failures[:20], "doc_chars": len(body)}


def verify_grouping(chunks: list) -> dict:
    """
    Distinct source paths must never share a citation group.

    This is the regression the citations module was written to fix: grouping
    on basename collapsed every __init__.py in a repository into one entry.
    psf/requests contains several, so the corpus genuinely exercises it.
    """
    group_to_sources: dict = defaultdict(set)
    for c in chunks:
        group_to_sources[citation_group_key(c)].add(c.metadata.get("source", ""))

    collisions = {str(k): sorted(v) for k, v in group_to_sources.items() if len(v) > 1}

    basenames: dict = defaultdict(set)
    for c in chunks:
        basenames[os.path.basename(c.metadata.get("source", ""))].add(
            c.metadata.get("source", ""))
    ambiguous = {b: sorted(v) for b, v in basenames.items() if len(v) > 1}

    return {
        "distinct_groups": len(group_to_sources),
        "collisions": collisions,
        "n_collisions": len(collisions),
        "shared_basenames_in_corpus": ambiguous,
        "n_shared_basenames": len(ambiguous),
    }


def main() -> int:
    from benchmark.build_corpus import CORPUS_DIR, DOCUMENTS, build_chunks
    from benchmark.validate_dataset import parent_texts

    chunks, _meta = build_chunks()
    parents = parent_texts()

    pdf = os.path.join(CORPUS_DIR, DOCUMENTS[0])
    txt = os.path.join(CORPUS_DIR, DOCUMENTS[1])
    docx = os.path.join(CORPUS_DIR, DOCUMENTS[2])

    report = {
        "repo_start_line": verify_repo(chunks, parents),
        "pdf_page_content": verify_pdf(chunks, pdf),
        "text_offset": verify_text(chunks, txt),
        "docx_offset": verify_docx(chunks, docx),
        "grouping": verify_grouping(chunks),
        "pptx": {"checked": 0,
                 "note": "no PPTX in the benchmark corpus; this check could "
                         "not be run and is reported as not measured"},
    }

    print("=" * 70)
    print("CITATION VERIFICATION")
    print("=" * 70)
    r = report["repo_start_line"]
    print(f"  repo start_line   : {r['checked']-r['mismatched']}/{r['checked']} "
          f"verified, {r['mismatched']} mismatched")
    p = report["pdf_page_content"]
    print(f"  PDF page content  : {p['checked']-p['mismatched']}/{p['checked']} "
          f"verified, {p['mismatched']} mismatched  ({p['n_pages']} pages)")
    t = report["text_offset"]
    print(f"  text offset       : {t['checked']-t['mismatched']}/{t['checked']} "
          f"verified, {t['mismatched']} mismatched")
    dx = report["docx_offset"]
    print(f"  docx start_index  : {dx['checked']-dx['mismatched']}/{dx['checked']} "
          f"verified, {dx['mismatched']} mismatched  ({dx['doc_chars']:,} chars)")
    g = report["grouping"]
    print(f"  citation grouping : {g['distinct_groups']} distinct groups, "
          f"{g['n_collisions']} collisions")
    print(f"                      {g['n_shared_basenames']} basename(s) shared by "
          f"multiple paths in the corpus:")
    for b, paths in list(g["shared_basenames_in_corpus"].items())[:6]:
        print(f"                        {b} -> {len(paths)} distinct paths")
    print(f"  PPTX              : not measured ({report['pptx']['note'][:44]}...)")

    for section in ("repo_start_line", "pdf_page_content", "text_offset",
                    "docx_offset"):
        for f in report[section]["failures"]:
            print(f"    FAILURE [{section}] {f}")

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nwritten: {REPORT_PATH}")

    total_bad = (report["repo_start_line"]["mismatched"]
                 + report["pdf_page_content"]["mismatched"]
                 + report["text_offset"]["mismatched"]
                 + report["docx_offset"]["mismatched"]
                 + report["grouping"]["n_collisions"])
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
