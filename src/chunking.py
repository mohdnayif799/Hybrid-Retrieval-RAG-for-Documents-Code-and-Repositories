"""
The single chunking entry point for every source type.

Split out of ``data_ingestion`` so it can be tested without importing
Streamlit, EasyOCR and PyMuPDF. ``data_ingestion`` re-exports
``chunk_documents``, so existing imports (including evaluate.py's) are
unaffected.

Why one function rather than a chunker per source
-------------------------------------------------
``hybrid_retrieval`` states the invariant it depends on: "one ingestion path,
one chunking policy, two views of it." The vector index and the BM25 index are
both built from whatever lands in Chroma, and ``chunk_key()`` identity assumes
chunks are produced consistently. Two chunking entry points would mean two
policies that could drift apart silently. So the dispatch happens *inside* this
function, and everything downstream stays identical.

Dispatch rule
-------------
Language-aware splitting applies only to documents carrying
``source_type == "repo"``, never on extension alone. That matters because
``.md`` and ``.txt`` exist in both worlds: an uploaded README must keep the
exact chunking it has today, while a repository README should be split on
Markdown headings. Extension cannot distinguish those two cases; source type
can.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.code_languages import language_for_path, separators_for
from src.repo_ingestion import SOURCE_TYPE_REPO

SOURCE_TYPE_UPLOAD = "upload"

# Document defaults — unchanged from the original implementation. Do not alter
# these without re-checking uploaded-document behaviour.
DOC_CHUNK_SIZE = 1000
DOC_CHUNK_OVERLAP = 200
DOC_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Code defaults. HEURISTIC, NOT MEASURED: code is denser than prose, so chunks
# are smaller, and language separators already cut at function and class
# boundaries, so less overlap is needed to avoid splitting mid-thought. These
# are named constants precisely so they can be tuned against a retrieval
# evaluation set later rather than being buried as magic numbers. No claim is
# made that they are optimal.
CODE_CHUNK_SIZE = 800
CODE_CHUNK_OVERLAP = 100

_splitter_cache: dict[tuple, RecursiveCharacterTextSplitter] = {}


def _get_splitter(language: Language | None, chunk_size: int,
                  chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Build (and memoise) a splitter. Falls back to the generic separators
    when this LangChain version has none for the language."""
    key = (language.value if language else None, chunk_size, chunk_overlap)
    if key in _splitter_cache:
        return _splitter_cache[key]

    separators = separators_for(language) if language else None
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators or DOC_SEPARATORS,
        add_start_index=True,
    )
    _splitter_cache[key] = splitter
    return splitter


def _plan(doc: Document, chunk_size: int, chunk_overlap: int):
    """Choose (splitter, is_repo) for one parent document."""
    if doc.metadata.get("source_type") == SOURCE_TYPE_REPO:
        language = language_for_path(doc.metadata.get("source", ""))
        return _get_splitter(language, CODE_CHUNK_SIZE, CODE_CHUNK_OVERLAP), True
    return _get_splitter(None, chunk_size, chunk_overlap), False


def start_line_of(text: str, start_index: int) -> int:
    """1-based line number of a character offset within text."""
    if start_index <= 0:
        return 1
    return text.count("\n", 0, start_index) + 1


def chunk_documents(documents: list, chunk_size: int = DOC_CHUNK_SIZE,
                    chunk_overlap: int = DOC_CHUNK_OVERLAP) -> list:
    """
    Split documents into overlapping chunks.

    Uploaded documents use the original generic splitter and settings.
    Repository documents use a language-aware splitter and gain a
    ``start_line`` so citations can point at ``src/module.py:142``.

    Splitting is done per parent document. RecursiveCharacterTextSplitter
    already processes documents independently, so this produces byte-identical
    output to the previous single batched call — there is a test asserting it.
    """
    chunks: list[Document] = []

    for doc in documents:
        splitter, is_repo = _plan(doc, chunk_size, chunk_overlap)
        pieces = splitter.split_documents([doc])

        for piece in pieces:
            if is_repo:
                piece.metadata["start_line"] = start_line_of(
                    doc.page_content, piece.metadata.get("start_index", 0)
                )
            else:
                # Make the source type explicit on new chunks. Chunks indexed
                # before this field existed simply lack it, and every consumer
                # treats a missing value as an upload.
                piece.metadata.setdefault("source_type", SOURCE_TYPE_UPLOAD)

        chunks.extend(pieces)

    n_repo = sum(1 for c in chunks if c.metadata.get("source_type") == SOURCE_TYPE_REPO)
    print(f"[INFO] Created {len(chunks)} chunks from {len(documents)} document(s) "
          f"({n_repo} from repository files).")
    return chunks
