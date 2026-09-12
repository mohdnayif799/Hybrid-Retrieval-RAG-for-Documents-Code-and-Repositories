"""
Chunking tests.

The most important test here is the first one: uploaded-document chunking must
be byte-identical to the pre-change implementation. Repository support is not
worth a silent regression in the behaviour that already worked.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking import (
    CODE_CHUNK_OVERLAP,
    CODE_CHUNK_SIZE,
    DOC_CHUNK_OVERLAP,
    DOC_CHUNK_SIZE,
    chunk_documents,
    start_line_of,
)
from src.repo_ingestion import RepoRef, RepoFile, repo_files_to_documents


def upload_doc(text, source="manual.pdf", page=0):
    return Document(page_content=text, metadata={"source": source, "page": page})


def repo_doc(text, path):
    return repo_files_to_documents([RepoFile(path=path, text=text)], RepoRef("o", "r"), "main")[0]


# ── Regression: existing document behaviour is untouched ──────────────────────

def test_document_chunking_is_byte_identical_to_the_previous_implementation():
    """Reproduces the original chunk_documents exactly and compares."""
    docs = [
        upload_doc("Lorem ipsum dolor sit amet. " * 200, source="a.pdf", page=3),
        upload_doc("Second document.\n\nWith paragraphs.\n" * 100, source="b.docx"),
    ]
    legacy = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""], add_start_index=True,
    ).split_documents(docs)

    new = chunk_documents(docs)

    assert len(new) == len(legacy)
    for a, b in zip(new, legacy):
        assert a.page_content == b.page_content
        assert a.metadata["source"] == b.metadata["source"]
        assert a.metadata.get("page") == b.metadata.get("page")
        assert a.metadata["start_index"] == b.metadata["start_index"]


def test_uploaded_markdown_keeps_generic_splitting():
    # .md maps to Language.MARKDOWN, but an UPLOADED .md must not change
    # behaviour — dispatch is on source_type, never on extension alone.
    doc = upload_doc("# Title\n\n" + ("body text. " * 300), source="notes.md")
    legacy = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""], add_start_index=True,
    ).split_documents([doc])
    assert [c.page_content for c in chunk_documents([doc])] == \
           [c.page_content for c in legacy]


def test_uploaded_chunks_are_labelled_as_uploads():
    chunks = chunk_documents([upload_doc("hello world " * 100)])
    assert all(c.metadata["source_type"] == "upload" for c in chunks)


def test_custom_chunk_size_still_honoured_for_documents():
    chunks = chunk_documents([upload_doc("word " * 500)], chunk_size=100, chunk_overlap=0)
    assert all(len(c.page_content) <= 100 for c in chunks)


# ── Repository chunking ───────────────────────────────────────────────────────

def test_python_is_split_on_definition_boundaries():
    code = "\n\n".join(f"def function_{i}():\n    " + "x = 1\n    " * 30 for i in range(6))
    chunks = chunk_documents([repo_doc(code, "src/mod.py")])
    assert len(chunks) > 1
    # Language separators include "\ndef ", so most chunks should begin at one.
    starts = sum(1 for c in chunks if c.page_content.lstrip().startswith("def "))
    assert starts >= len(chunks) // 2


def test_repo_chunks_use_code_chunk_size():
    code = "line = 1\n" * 500
    chunks = chunk_documents([repo_doc(code, "src/mod.py")])
    assert all(len(c.page_content) <= CODE_CHUNK_SIZE for c in chunks)
    assert CODE_CHUNK_SIZE != DOC_CHUNK_SIZE and CODE_CHUNK_OVERLAP != DOC_CHUNK_OVERLAP


def test_unknown_language_falls_back_to_generic_splitter():
    chunks = chunk_documents([repo_doc("key = value\n" * 300, "config.toml")])
    assert chunks and all(c.metadata["source_type"] == "repo" for c in chunks)


def test_repo_metadata_survives_chunking():
    chunks = chunk_documents([repo_doc("def f():\n    return 1\n" * 80, "src/a/b.py")])
    for c in chunks:
        assert c.metadata["source"] == "src/a/b.py"
        assert c.metadata["repo"] == "o/r"
        assert c.metadata["ref"] == "main"
        assert c.metadata["source_type"] == "repo"
        assert "start_index" in c.metadata and "start_line" in c.metadata


# ── Line numbers ──────────────────────────────────────────────────────────────

def test_start_line_of_basic_offsets():
    text = "alpha\nbravo\ncharlie\n"
    assert start_line_of(text, 0) == 1
    assert start_line_of(text, 6) == 2      # start of "bravo"
    assert start_line_of(text, 12) == 3     # start of "charlie"


def test_start_line_is_one_based_and_never_zero():
    assert start_line_of("anything", -5) == 1
    assert start_line_of("", 0) == 1


def test_line_numbers_are_correct_and_increasing():
    lines = [f"# comment line {i}" for i in range(400)]
    code = "\n".join(lines)
    chunks = chunk_documents([repo_doc(code, "src/mod.py")])

    numbers = [c.metadata["start_line"] for c in chunks]
    assert numbers[0] == 1
    assert numbers == sorted(numbers)

    # Each reported line must actually contain that chunk's first line of text.
    for c in chunks:
        first = c.page_content.splitlines()[0].strip()
        actual = lines[c.metadata["start_line"] - 1].strip()
        assert actual.startswith(first[:20]) or first.startswith(actual[:20])


def test_uploaded_chunks_get_no_start_line():
    chunks = chunk_documents([upload_doc("text " * 400)])
    assert all("start_line" not in c.metadata for c in chunks)


# ── Mixed corpus ──────────────────────────────────────────────────────────────

def test_documents_and_repository_files_chunk_together_in_one_call():
    mixed = [
        upload_doc("Prose about refunds. " * 100, source="policy.pdf", page=2),
        repo_doc("def refund():\n    pass\n" * 60, "src/billing.py"),
    ]
    chunks = chunk_documents(mixed)
    kinds = {c.metadata["source_type"] for c in chunks}
    assert kinds == {"upload", "repo"}
    assert all("start_line" in c.metadata
               for c in chunks if c.metadata["source_type"] == "repo")
