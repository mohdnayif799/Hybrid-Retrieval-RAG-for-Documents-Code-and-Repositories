"""Citation formatting tests — pure, no Streamlit."""

from langchain_core.documents import Document

import re

import pytest

from src.citations import (
    build_inline_citations,
    escape_markdown,
    citation_display_name,
    citation_group_key,
    doc_type_label,
    format_location,
    format_reference,
    retrieval_badge,
)


def render(md: str) -> str:
    """
    Markdown source -> the plain text a user actually sees.

    Asserting on the raw string is what let the __init__.py bug ship: the
    source contained the path, but the renderer ate the underscores.
    """
    markdown = pytest.importorskip("markdown")
    return re.sub(r"<[^>]+>", "", markdown.markdown(md))


def repo_chunk(source, start_line=1, repo="psf/requests", retrieval=None):
    meta = {"source": source, "source_type": "repo", "repo": repo,
            "ref": "main", "page": 0, "start_line": start_line, "start_index": 0}
    if retrieval:
        meta["retrieval"] = retrieval
    return Document(page_content="code", metadata=meta)


def upload_chunk(source, page=0, start_index=0, source_type="upload"):
    return Document(page_content="text", metadata={
        "source": source, "page": page, "start_index": start_index,
        "source_type": source_type,
    })


# ── Existing document behaviour must not change ───────────────────────────────

def test_pdf_location_unchanged():
    assert format_location(upload_chunk("manual.pdf", page=4), 1) == "Page 5"


def test_pptx_location_unchanged():
    assert format_location(upload_chunk("deck.pptx", page=2), 1) == "Slide 3"


def test_text_location_unchanged():
    assert format_location(upload_chunk("notes.txt", start_index=1200), 1) == "~char 1,200"
    assert format_location(upload_chunk("notes.md", start_index=45), 1) == "~char 45"


def test_unknown_upload_type_falls_back_to_chunk_index():
    assert format_location(upload_chunk("mystery.xyz"), 7) == "Chunk 7"


def test_upload_labels_unchanged():
    assert doc_type_label("manual.pdf") == "\U0001f4d5 PDF"
    assert doc_type_label("notes.md") == "\U0001f4dd Markdown"
    assert doc_type_label("thing.xyz") == "\U0001f4c4 File"


def test_chunks_without_source_type_are_treated_as_uploads():
    """Chunks indexed before source_type existed must still render."""
    legacy = Document(page_content="x", metadata={"source": "old.pdf", "page": 0})
    assert format_location(legacy, 1) == "Page 1"
    assert citation_group_key(legacy)[0] == "upload"


# ── Repository citations ──────────────────────────────────────────────────────

def test_repo_reference_is_path_colon_line():
    assert format_reference(repo_chunk("src/rag_chain.py", 142), 1) == "src/rag_chain.py:142"


def test_repo_location_is_a_line_marker():
    assert format_location(repo_chunk("src/app.py", 88), 1) == "L88"


def test_repo_display_name_keeps_the_full_relative_path():
    name = citation_display_name(repo_chunk("src/deep/nested/mod.py"))
    assert "src/deep/nested/mod.py" in name
    assert name.startswith("psf/requests")


def test_repo_type_labels_are_language_aware():
    assert doc_type_label("src/a.py", "repo") == "\U0001f40d Python"
    assert doc_type_label("src/a.ts", "repo") == "\U0001f537 TypeScript"
    assert doc_type_label("src/a.weird", "repo") == "\U0001f4c1 Repository file"


def test_repo_markdown_and_uploaded_markdown_are_distinguishable():
    # Same extension, different provenance — must not share a citation group.
    a = citation_group_key(repo_chunk("README.md"))
    b = citation_group_key(upload_chunk("README.md"))
    assert a != b


# ── The grouping bug ──────────────────────────────────────────────────────────

def test_same_basename_in_different_directories_does_not_merge():
    docs = [
        repo_chunk("src/a/__init__.py", 1),
        repo_chunk("src/b/__init__.py", 1),
        repo_chunk("src/c/__init__.py", 1),
    ]
    assert len({citation_group_key(d) for d in docs}) == 3
    raw = build_inline_citations(docs)
    assert len(raw.split("\n\n")) == 3
    # Assert on the RENDERED text, not the markdown source.
    shown = render(raw)
    assert "src/a/__init__.py" in shown
    assert "src/b/__init__.py" in shown
    assert "src/c/__init__.py" in shown


def test_same_path_in_different_repositories_does_not_merge():
    docs = [repo_chunk("src/app.py", 1, repo="org/one"),
            repo_chunk("src/app.py", 1, repo="org/two")]
    assert len({citation_group_key(d) for d in docs}) == 2


def test_multiple_lines_from_one_file_group_into_one_entry():
    docs = [repo_chunk("src/app.py", 10), repo_chunk("src/app.py", 90)]
    rendered = build_inline_citations(docs)
    assert rendered.count("src/app.py") == 1
    assert "L10, L90" in rendered


def test_uploaded_files_still_group_by_basename():
    docs = [upload_chunk("manual.pdf", page=0), upload_chunk("manual.pdf", page=4)]
    rendered = build_inline_citations(docs)
    assert rendered.count("manual.pdf") == 1
    assert "Page 1, Page 5" in rendered


def test_mixed_document_and_repository_citations_render_together():
    rendered = build_inline_citations([
        upload_chunk("policy.pdf", page=2),
        repo_chunk("src/billing.py", 42),
    ])
    assert "policy.pdf" in rendered and "Page 3" in rendered
    assert "src/billing.py" in rendered and "L42" in rendered


# ── Retrieval provenance badge ────────────────────────────────────────────────

def test_retrieval_badge_variants():
    assert "semantic + keyword" in retrieval_badge(repo_chunk("a.py", retrieval="vector+bm25"))
    assert "keyword" in retrieval_badge(repo_chunk("a.py", retrieval="bm25"))
    assert retrieval_badge(repo_chunk("a.py")) == ""


def test_empty_citation_list_is_empty_string():
    assert build_inline_citations([]) == ""


# ── Markdown safety (regression: __init__.py rendered as init.py) ─────────────

def test_dunder_paths_survive_the_markdown_renderer():
    # CommonMark reads __init__ at a word boundary as strong emphasis, so the
    # citation displayed "starter_repo/init.py" — a path that does not exist.
    shown = render(build_inline_citations([repo_chunk("starter_repo/__init__.py", 1)]))
    assert "starter_repo/__init__.py" in shown
    assert "starter_repo/init.py" not in shown


def test_dunder_main_and_multi_underscore_paths_survive():
    for path in ("pkg/__main__.py", "tests/test_data_loader.py", "a/__about__.py"):
        shown = render(build_inline_citations([repo_chunk(path, 3)]))
        assert path in shown, f"{path} was mangled by the renderer"


def test_uploaded_filenames_with_underscores_survive():
    shown = render(build_inline_citations([upload_chunk("My_Resume_ATS.docx", page=0)]))
    assert "My_Resume_ATS.docx" in shown


def test_escape_markdown_leaves_plain_text_alone():
    assert escape_markdown("src/app.py") == "src/app.py"


# ── Location ordering ─────────────────────────────────────────────────────────

def test_repo_locations_are_listed_in_file_order():
    # Retrieval rank produced "L89, L58, L1, L79", which reads like a bug.
    docs = [repo_chunk("README.md", n) for n in (89, 58, 1, 79)]
    assert "L1, L58, L79, L89" in build_inline_citations(docs)


def test_pdf_pages_are_listed_in_page_order():
    docs = [upload_chunk("manual.pdf", page=p) for p in (7, 1, 4)]
    assert "Page 2, Page 5, Page 8" in build_inline_citations(docs)


def test_duplicate_locations_are_collapsed():
    docs = [repo_chunk("src/app.py", 10), repo_chunk("src/app.py", 10)]
    assert build_inline_citations(docs).count("L10") == 1
