"""
Citation rendering for retrieved chunks.

Extracted from app.py because importing app.py runs the entire Streamlit
script, which makes these helpers untestable in place. They are pure string
functions over Document metadata, so they belong outside the UI anyway.

The bug this fixes
------------------
Citations were grouped by ``os.path.basename(source)``. For uploaded files
that is fine — names are unique. For a repository it is wrong: every
``__init__.py``, ``utils.py`` and ``index.ts`` in the tree collapses into a
single citation group, so the user is told the answer came from "utils.py"
with no way to tell which of the eleven files that is. Grouping now uses the
full relative path plus the source type, so a repository ``README.md`` also
stays distinct from an uploaded one.

Backwards compatibility: chunks indexed before ``source_type`` existed simply
lack the field, and every function here treats a missing value as an upload.
"""

from __future__ import annotations

import os

from src.repo_ingestion import SOURCE_TYPE_REPO

UNKNOWN_SOURCE = "unknown"

# Characters that Streamlit's markdown renderer would interpret inside a file
# path. The one that actually bites is the double underscore: a Python package
# path like ``starter_repo/__init__.py`` has ``__init__`` sitting at a word
# boundary (preceded by "/"), which CommonMark reads as strong emphasis, so the
# citation rendered as "starter_repo/init.py" — a path that does not exist.
# Single intraword underscores (``plot_data.py``) are safe under CommonMark,
# but escaping uniformly is cheaper than reasoning about flanking rules.
_MD_SPECIAL = ("\\", "`", "*", "_", "[", "]", "~")


def escape_markdown(text: str) -> str:
    """Escape markdown control characters so a file path renders literally."""
    for char in _MD_SPECIAL:
        text = text.replace(char, "\\" + char)
    return text

_DOC_EMOJI = {
    ".pdf": "\U0001f4d5", ".pptx": "\U0001f4ca", ".docx": "\U0001f4c4",
    ".doc": "\U0001f4c4", ".txt": "\U0001f4dd", ".md": "\U0001f4dd",
}

_DOC_TYPE_NAMES = {
    ".pdf": "\U0001f4d5 PDF", ".pptx": "\U0001f4ca Presentation",
    ".docx": "\U0001f4c4 Document", ".doc": "\U0001f4c4 Document",
    ".txt": "\U0001f4dd Text file", ".md": "\U0001f4dd Markdown",
}

_CODE_TYPE_NAMES = {
    ".py": "\U0001f40d Python", ".js": "\U0001f7e8 JavaScript",
    ".jsx": "\U0001f7e8 JavaScript", ".mjs": "\U0001f7e8 JavaScript",
    ".cjs": "\U0001f7e8 JavaScript", ".ts": "\U0001f537 TypeScript",
    ".tsx": "\U0001f537 TypeScript", ".java": "\u2615 Java",
    ".go": "\U0001f439 Go", ".rs": "\U0001f980 Rust", ".rb": "\U0001f48e Ruby",
    ".php": "\U0001f418 PHP", ".cs": "\U0001f7ea C#", ".kt": "\U0001f7e3 Kotlin",
    ".swift": "\U0001f426 Swift", ".c": "\u2699\ufe0f C", ".h": "\u2699\ufe0f C header",
    ".cpp": "\u2699\ufe0f C++", ".cc": "\u2699\ufe0f C++", ".hpp": "\u2699\ufe0f C++",
    ".sh": "\U0001f41a Shell", ".bash": "\U0001f41a Shell",
    ".sql": "\U0001f5c3\ufe0f SQL", ".html": "\U0001f310 HTML",
    ".css": "\U0001f3a8 CSS", ".scss": "\U0001f3a8 CSS",
    ".json": "\U0001f9fe JSON", ".yaml": "\U0001f9fe YAML", ".yml": "\U0001f9fe YAML",
    ".toml": "\U0001f9fe TOML", ".md": "\U0001f4dd Markdown",
}

_CODE_FALLBACK = "\U0001f4c1 Repository file"


def is_repo_chunk(doc) -> bool:
    return doc.metadata.get("source_type") == SOURCE_TYPE_REPO


def citation_group_key(doc) -> tuple:
    """
    Identity for grouping citations.

    Repository chunks key on (type, repo, full relative path) so identically
    named files in different directories — or in different repositories — never
    merge. Uploaded files keep the historical basename grouping.
    """
    source = doc.metadata.get("source", UNKNOWN_SOURCE)
    if is_repo_chunk(doc):
        return (SOURCE_TYPE_REPO, doc.metadata.get("repo", ""), source)
    return ("upload", "", os.path.basename(source))


def citation_display_name(doc) -> str:
    """Human-readable name for a chunk's file."""
    source = doc.metadata.get("source", UNKNOWN_SOURCE)
    if is_repo_chunk(doc):
        repo = doc.metadata.get("repo")
        return f"{repo} · {source}" if repo else source
    return os.path.basename(source)


def format_location(doc, chunk_index: int) -> str:
    """Where inside its file this chunk sits, for use under a file heading."""
    source = doc.metadata.get("source", "")
    ext = os.path.splitext(source)[1].lower()

    if is_repo_chunk(doc):
        line = doc.metadata.get("start_line")
        return f"L{int(line)}" if line is not None else f"Chunk {chunk_index}"

    page = doc.metadata.get("page", None)
    if ext == ".pdf" and page is not None:
        return f"Page {int(page) + 1}"
    if ext == ".pptx" and page is not None:
        return f"Slide {int(page) + 1}"
    if ext in {".txt", ".md"}:
        start = doc.metadata.get("start_index", None)
        return f"~char {int(start):,}" if start is not None else f"Chunk {chunk_index}"
    return f"Chunk {chunk_index}"


def citation_sort_key(doc, chunk_index: int) -> int:
    """
    Position of a chunk within its file, for ordering locations in a citation.

    Without this, a file cited three times lists its locations in retrieval-rank
    order — "L89, L58, L1" — which reads like a bug even though it is not.
    """
    if is_repo_chunk(doc):
        return int(doc.metadata.get("start_line") or 0)
    page = doc.metadata.get("page")
    if page is not None:
        ext = os.path.splitext(doc.metadata.get("source", ""))[1].lower()
        if ext in {".pdf", ".pptx"}:
            return int(page)
    start = doc.metadata.get("start_index")
    return int(start) if start is not None else chunk_index


def format_reference(doc, chunk_index: int) -> str:
    """
    Fully qualified reference, e.g. ``src/rag_chain.py:142`` for code or
    ``manual.pdf`` for an upload. Used as the heading of each source chunk.
    """
    if is_repo_chunk(doc):
        source = doc.metadata.get("source", UNKNOWN_SOURCE)
        line = doc.metadata.get("start_line")
        return f"{source}:{int(line)}" if line is not None else source
    return os.path.basename(doc.metadata.get("source", UNKNOWN_SOURCE))


def doc_type_label(source: str, source_type: str | None = None) -> str:
    """Emoji + type name. Code types are only used for repository sources so an
    uploaded .md keeps its original 'Markdown' label."""
    ext = os.path.splitext(source)[1].lower()
    if source_type == SOURCE_TYPE_REPO:
        return _CODE_TYPE_NAMES.get(ext, _CODE_FALLBACK)
    return _DOC_TYPE_NAMES.get(ext, "\U0001f4c4 File")


def doc_type_label_for(doc) -> str:
    return doc_type_label(doc.metadata.get("source", ""), doc.metadata.get("source_type"))


def retrieval_badge(doc) -> str:
    """Which retriever(s) surfaced a chunk. Set by the RRF layer; absent when
    retrieval fell back to vector-only, in which case nothing is rendered."""
    origin = doc.metadata.get("retrieval")
    if not origin:
        return ""
    label = {
        "vector": "\U0001f9e0 semantic",
        "bm25": "\U0001f524 keyword",
        "vector+bm25": "\u2b50 semantic + keyword",
        "bm25+vector": "\u2b50 semantic + keyword",
    }.get(origin, origin)
    return f" · *{label}*"


def build_inline_citations(source_docs: list) -> str:
    """Grouped 'Sources:' block shown beneath an answer."""
    grouped: dict = {}
    for i, doc in enumerate(source_docs, 1):
        key = citation_group_key(doc)
        entry = grouped.setdefault(key, {"name": citation_display_name(doc),
                                         "doc": doc, "locs": []})
        entry["locs"].append((citation_sort_key(doc, i), format_location(doc, i)))

    lines = []
    for entry in grouped.values():
        doc = entry["doc"]
        source = doc.metadata.get("source", "")
        ext = os.path.splitext(source)[1].lower()
        if is_repo_chunk(doc):
            emoji = _CODE_TYPE_NAMES.get(ext, _CODE_FALLBACK).split(" ")[0]
        else:
            emoji = _DOC_EMOJI.get(ext, "\U0001f4c4")

        # Locations in file order, de-duplicated; name escaped so paths survive
        # the markdown renderer.
        seen, ordered = set(), []
        for _, label in sorted(entry["locs"], key=lambda pair: pair[0]):
            if label not in seen:
                seen.add(label)
                ordered.append(label)
        lines.append(
            f"{emoji} **{escape_markdown(entry['name'])}** \u2014 {', '.join(ordered)}"
        )
    return "\n\n".join(lines)
