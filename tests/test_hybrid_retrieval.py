"""
Tests for the hybrid retrieval layer.

Runs with no Streamlit runtime, no Chroma, no embedding model and no API key:
src/hybrid_retrieval.py imports vector_store lazily, so the ranking logic is
testable in isolation. Requires only langchain-core, rank_bm25 and pytest.

    python -m pytest tests/ -v
"""

import pytest
from langchain_core.documents import Document

from src.hybrid_retrieval import (
    BM25Index,
    HybridRetriever,
    chunk_key,
    is_repo_readme,
    reciprocal_rank_fusion,
    tokenize,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_doc(text: str, source: str = "manual.pdf", page: int = 0, start: int = 0) -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "page": page, "start_index": start},
    )


@pytest.fixture
def corpus():
    return [
        make_doc("The refund policy is described in Section 4.2 of this agreement.", start=0),
        make_doc("Customers may cancel their subscription at any time.", start=1000),
        make_doc("Photosynthesis converts light energy into chemical energy.", start=2000),
        make_doc("Error code CVE-2024-3094 affects the xz compression library.", start=3000),
        make_doc("Termination of the agreement requires thirty days written notice.", start=4000),
    ]


class StubVectorRetriever:
    """Stands in for Chroma's MMR retriever: returns a fixed ranked list."""

    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        return list(self.docs)


# ── Tokenisation ──────────────────────────────────────────────────────────────

def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Section 4.2, Refunds!") == ["section", "4", "2", "refunds"]


def test_tokenize_removes_stopwords():
    assert tokenize("What is the refund policy?") == ["refund", "policy"]


def test_tokenize_splits_hyphenated_terms_consistently():
    # Doc and query must agree, which is what makes "gpt 4" match "GPT-4".
    assert tokenize("GPT-4") == tokenize("gpt 4") == ["gpt", "4"]


def test_tokenize_all_stopwords_yields_nothing():
    assert tokenize("what is it about?") == []


# ── Chunk identity ────────────────────────────────────────────────────────────

def test_chunk_key_matches_for_same_chunk_from_either_retriever():
    a = make_doc("same text", start=500)
    b = make_doc("same text", start=500)
    assert chunk_key(a) == chunk_key(b)


def test_chunk_key_distinguishes_different_offsets():
    assert chunk_key(make_doc("x", start=0)) != chunk_key(make_doc("x", start=1000))


def test_chunk_key_falls_back_to_content_hash_without_start_index():
    legacy = Document(page_content="old chunk", metadata={"source": "a.pdf"})
    assert chunk_key(legacy)[0] == "hash"


# ── BM25 ──────────────────────────────────────────────────────────────────────

def test_bm25_finds_exact_rare_token(corpus):
    index = BM25Index(corpus)
    hits = index.search("CVE-2024-3094", k=3)
    assert hits, "BM25 returned nothing for an exact identifier"
    assert "CVE-2024-3094" in hits[0][0].page_content


def test_bm25_returns_only_chunks_sharing_a_query_term(corpus):
    index = BM25Index(corpus)
    hits = index.search("photosynthesis", k=5)
    # Only one chunk shares that term; the rest must not be padded in.
    assert len(hits) == 1
    assert "Photosynthesis" in hits[0][0].page_content


def test_bm25_finds_matches_in_a_small_corpus():
    # Regression: Okapi IDF is negative for terms present in most of a small
    # corpus, so filtering on `score > 0` used to return nothing here — the
    # common case of a chat holding one short upload.
    docs = [
        make_doc("The refund policy allows returns within 30 days.", start=0),
        make_doc("The refund process takes five business days.", start=1000),
    ]
    hits = BM25Index(docs).search("refund", k=5)
    assert len(hits) == 2, "small-corpus lexical matches were dropped"


def test_bm25_respects_k(corpus):
    index = BM25Index(corpus)
    assert len(index.search("agreement", k=1)) <= 1


def test_bm25_empty_query_returns_nothing(corpus):
    index = BM25Index(corpus)
    assert index.search("what is it?", k=5) == []
    assert index.search("", k=5) == []


def test_bm25_empty_corpus_is_safe():
    index = BM25Index([])
    assert not index.ready
    assert index.search("anything", k=5) == []


def test_bm25_skips_untokenizable_chunks_but_keeps_the_rest():
    docs = [
        make_doc("--- === ...", source="scan.pdf", start=0),      # OCR artefact
        make_doc("", source="deck.pptx", start=1000),             # blank slide
        make_doc("Revenue grew 12 percent in Q3.", start=2000),
    ]
    index = BM25Index(docs)
    assert index.skipped_empty == 2
    assert len(index) == 1
    hits = index.search("Q3 revenue", k=5)
    assert len(hits) == 1 and "Q3" in hits[0][0].page_content


def test_bm25_corpus_of_only_untokenizable_chunks_does_not_raise():
    # Regression: rank_bm25 divides by len(idf) and raises ZeroDivisionError
    # when every document is empty — e.g. a chat whose only upload is a scan
    # that OCR-ed to punctuation.
    index = BM25Index([make_doc("..."), make_doc("")])
    assert not index.ready
    assert index.search("anything", k=3) == []


def test_bm25_scores_are_descending(corpus):
    index = BM25Index(corpus)
    scores = [s for _, s in index.search("agreement notice termination", k=5)]
    assert scores == sorted(scores, reverse=True)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def test_rrf_promotes_documents_found_by_both_retrievers():
    shared = make_doc("found by both", start=100)
    dense_only = make_doc("dense only", start=200)
    sparse_only = make_doc("sparse only", start=300)

    fused = reciprocal_rank_fusion(
        {"vector": [dense_only, shared], "bm25": [sparse_only, shared]},
        {"vector": 0.5, "bm25": 0.5},
        top_k=3,
    )
    assert fused[0].page_content == "found by both"
    assert fused[0].metadata["retrieval"] == "vector+bm25"


def test_rrf_preserves_order_of_a_single_list():
    docs = [make_doc(f"chunk {i}", start=i * 100) for i in range(4)]
    fused = reciprocal_rank_fusion(
        {"vector": docs, "bm25": []}, {"vector": 0.5, "bm25": 0.5}, top_k=4
    )
    assert [d.page_content for d in fused] == [d.page_content for d in docs]


def test_rrf_deduplicates_across_lists():
    doc = make_doc("only once", start=42)
    fused = reciprocal_rank_fusion(
        {"vector": [doc], "bm25": [doc]}, {"vector": 0.5, "bm25": 0.5}, top_k=5
    )
    assert len(fused) == 1


def test_rrf_respects_top_k():
    docs = [make_doc(f"c{i}", start=i * 100) for i in range(10)]
    assert len(reciprocal_rank_fusion({"vector": docs}, {"vector": 1.0}, top_k=5)) == 5


def test_rrf_preserves_citation_metadata():
    doc = make_doc("cite me", source="report.pdf", page=7, start=900)
    fused = reciprocal_rank_fusion({"vector": [doc]}, {"vector": 1.0}, top_k=1)
    assert fused[0].metadata["source"] == "report.pdf"
    assert fused[0].metadata["page"] == 7
    assert fused[0].metadata["start_index"] == 900


def test_rrf_does_not_mutate_input_documents():
    # The BM25 index caches its Documents for the whole session; in-place
    # annotation would leak one query's provenance into the next.
    doc = make_doc("cached chunk", start=1)
    reciprocal_rank_fusion({"vector": [doc]}, {"vector": 1.0}, top_k=1)
    assert "retrieval" not in doc.metadata
    assert "rrf_score" not in doc.metadata


def test_rrf_weight_breaks_ties_toward_the_heavier_retriever():
    dense = make_doc("dense pick", start=1)
    sparse = make_doc("sparse pick", start=2)
    fused = reciprocal_rank_fusion(
        {"vector": [dense], "bm25": [sparse]}, {"vector": 0.9, "bm25": 0.1}, top_k=2
    )
    assert fused[0].page_content == "dense pick"


# ── HybridRetriever ───────────────────────────────────────────────────────────

def test_hybrid_returns_final_k_documents(corpus):
    retriever = HybridRetriever(
        vector_retriever=StubVectorRetriever(corpus),
        bm25_index=BM25Index(corpus),
        final_k=3,
    )
    assert len(retriever.invoke("refund policy")) == 3


def test_hybrid_surfaces_lexical_hit_the_vector_side_missed(corpus):
    # Dense retriever returns everything EXCEPT the chunk with the identifier.
    dense_docs = [d for d in corpus if "CVE" not in d.page_content]
    retriever = HybridRetriever(
        vector_retriever=StubVectorRetriever(dense_docs),
        bm25_index=BM25Index(corpus),
        final_k=5,
    )
    results = retriever.invoke("CVE-2024-3094")
    assert any("CVE-2024-3094" in d.page_content for d in results)
    hit = next(d for d in results if "CVE-2024-3094" in d.page_content)
    assert hit.metadata["retrieval"] == "bm25"


def test_hybrid_degrades_to_vector_only_without_bm25(corpus):
    retriever = HybridRetriever(
        vector_retriever=StubVectorRetriever(corpus), bm25_index=None, final_k=5
    )
    results = retriever.invoke("anything at all")
    assert [d.page_content for d in results] == [d.page_content for d in corpus]
    assert all(d.metadata["retrieval"] == "vector" for d in results)


def test_hybrid_handles_all_stopword_query(corpus):
    # BM25 contributes nothing; must not crash or return an arbitrary slice.
    retriever = HybridRetriever(
        vector_retriever=StubVectorRetriever(corpus),
        bm25_index=BM25Index(corpus),
        final_k=5,
    )
    results = retriever.invoke("what is it about?")
    assert all(d.metadata["retrieval"] == "vector" for d in results)


def test_hybrid_passes_the_rewritten_query_through(corpus):
    stub = StubVectorRetriever(corpus)
    retriever = HybridRetriever(vector_retriever=stub, bm25_index=BM25Index(corpus))
    retriever.invoke("standalone rewritten question")
    assert stub.calls == ["standalone rewritten question"]


def test_hybrid_output_is_deterministic(corpus):
    retriever = HybridRetriever(
        vector_retriever=StubVectorRetriever(corpus),
        bm25_index=BM25Index(corpus),
        final_k=5,
    )
    a = [d.page_content for d in retriever.invoke("agreement termination notice")]
    b = [d.page_content for d in retriever.invoke("agreement termination notice")]
    assert a == b


# ── Code-aware tokenisation (added for repository ingestion) ──────────────────

def test_tokenize_splits_camel_case():
    assert tokenize("buildHybridRetriever") == ["build", "hybrid", "retriever"]


def test_tokenize_splits_pascal_case():
    assert tokenize("UserAuthService") == ["user", "auth", "service"]


def test_tokenize_splits_acronym_boundaries():
    # HTTPServer must not become ["httpserver"] or ["h","t","t","p","server"].
    assert tokenize("HTTPServerConfig") == ["http", "server", "config"]


def test_camel_case_identifier_is_reachable_from_natural_language():
    # The regression that motivated the change: before splitting, a JS/Java
    # codebase produced zero lexical overlap for any decomposed query, and
    # hybrid retrieval silently fell back to vector-only.
    doc = set(tokenize("function buildHybridRetriever(chromaDir) { return new Retriever(); }"))
    query = set(tokenize("build hybrid retriever"))
    assert doc & query == {"build", "hybrid", "retriever"}


def test_exact_identifier_paste_still_matches_after_splitting():
    doc = set(tokenize("class UserAuthService { validateAccessToken() {} }"))
    assert doc & set(tokenize("validateAccessToken")) == {"validate", "access", "token"}


def test_snake_case_behaviour_is_unchanged():
    assert tokenize("build_hybrid_retriever") == ["build", "hybrid", "retriever"]


def test_camel_split_does_not_disturb_prose_or_identifiers():
    # Guards the existing document corpus: all-caps and hyphenated tokens have
    # no lower->upper boundary, so they must tokenize exactly as before.
    assert tokenize("GPT-4") == ["gpt", "4"]
    assert tokenize("CVE-2024-3094") == ["cve", "2024", "3094"]
    assert tokenize("Section 4.2, Refunds!") == ["section", "4", "2", "refunds"]
    assert tokenize("ISO 27001") == ["iso", "27001"]


def test_language_keywords_remain_stopwords_by_design():
    # Documents the deliberate decision in _STOPWORDS: keeping these out keeps
    # the candidacy filter selective. "for loop" still works via "loop".
    assert tokenize("for") == []
    assert tokenize("for loop") == ["loop"]
    # Discriminative code keywords stay searchable.
    for kw in ("class", "def", "function", "return", "import", "async", "await", "self"):
        assert tokenize(kw) == [kw], f"{kw} should be searchable"


def test_bm25_finds_camel_case_definition_in_a_code_corpus():
    docs = [
        make_doc("export function buildHybridRetriever(chromaDir) { return r; }",
                 source="src/retrieval.ts", start=0),
        make_doc("export function renderSidebar(props) { return null; }",
                 source="src/ui.ts", start=1000),
        make_doc("The quarterly refund policy is described in Section 4.2.",
                 source="policy.pdf", start=2000),
    ]
    hits = BM25Index(docs).search("build hybrid retriever", k=3)
    assert hits, "camelCase definition was not reachable lexically"
    assert "buildHybridRetriever" in hits[0][0].page_content


# ── README-crowding backstop (real production bug: BM25 term-frequency bias) ──

def readme_doc(text="This project does X.", start=0):
    return Document(page_content=text, metadata={
        "source": "README.md", "source_type": "repo", "repo": "o/r",
        "ref": "main", "page": 0, "start_index": start,
    })


def impl_doc(text, source="src/impl.py", start=0):
    return Document(page_content=text, metadata={
        "source": source, "source_type": "repo", "repo": "o/r",
        "ref": "main", "page": 0, "start_index": start,
    })


def test_search_within_finds_a_document_the_normal_top_k_would_exclude():
    # Background documents that don't mention "repository" at all -- without
    # these, every document shares the term and BM25's IDF collapses to near
    # zero for it, which is exactly why the ORIGINAL version of this test
    # failed: it accidentally erased the real skew it was meant to reproduce.
    background = [impl_doc(f"unrelated helper function number {i}", start=100 + i) for i in range(5)]
    crowding = [impl_doc("repository repository repository handler", start=i) for i in range(10)]
    docs = background + crowding + [readme_doc("This repository is a small config loader.", start=999)]
    index = BM25Index(docs)

    assert all(d.metadata["source"] != "README.md" for d, _ in index.search("repository", k=5))

    hits = index.search_within("repository", is_repo_readme, k=1)
    assert hits and hits[0][0].metadata["source"] == "README.md"


def test_search_within_respects_the_zero_relevance_guard():
    docs = [impl_doc("x"), readme_doc("completely unrelated content about gardening")]
    index = BM25Index(docs)
    assert index.search_within("kubernetes deployment", is_repo_readme, k=1) == []


def test_hybrid_surfaces_readme_when_bm25_would_otherwise_crowd_it_out():
    # Reproduces the measured production failure: repo_ingestion.py-style
    # density crowds README off the standard candidate list entirely.
    crowding = [impl_doc(f"repository repository repository handler {i}", start=i) for i in range(10)]
    readme = readme_doc("This repository implements a hybrid retrieval RAG system.", start=999)
    all_docs = crowding + [readme]

    retriever = HybridRetriever(
        vector_retriever=StubVectorRetriever(crowding),  # dense ALSO misses it — worst case
        bm25_index=BM25Index(all_docs),
        final_k=5,
    )
    results = retriever.invoke("what is this repository about")
    assert any(is_repo_readme(d) for d in results)
    boosted = next(d for d in results if is_repo_readme(d))
    assert boosted.metadata["retrieval"] == "readme_boost"


def test_readme_boost_does_not_fire_when_readme_already_present():
    docs = [readme_doc("This repository does the thing.", start=0),
            impl_doc("some code", start=1)]
    retriever = HybridRetriever(
        vector_retriever=StubVectorRetriever(docs), bm25_index=BM25Index(docs), final_k=5,
    )
    results = retriever.invoke("this repository")
    origins = {d.metadata.get("retrieval") for d in results if is_repo_readme(d)}
    assert "readme_boost" not in origins  # earned its place naturally, not injected


def test_readme_boost_never_forces_in_zero_relevance_content():
    crowding = [impl_doc(f"repository handler config {i}", start=i) for i in range(10)]
    readme = readme_doc("Completely unrelated gardening notes.", start=999)
    retriever = HybridRetriever(
        vector_retriever=StubVectorRetriever(crowding),
        bm25_index=BM25Index(crowding + [readme]),
        final_k=5,
    )
    results = retriever.invoke("repository configuration handler")
    assert not any(is_repo_readme(d) for d in results)  # no shared vocabulary — must not be forced in


def test_nested_readme_is_not_treated_as_the_project_readme():
    nested = impl_doc("nested docs", source="vendor/submodule/README.md", start=0)
    assert not is_repo_readme(nested)


def test_uploaded_file_named_readme_is_not_affected():
    uploaded = Document(page_content="x", metadata={"source": "README.md", "source_type": "upload", "page": 0})
    assert not is_repo_readme(uploaded)


def test_first_matching_returns_the_lowest_start_index_chunk():
    docs = [readme_doc("middle section", start=500), readme_doc("opening", start=0),
            readme_doc("later section", start=1200)]
    index = BM25Index(docs)
    found = index.first_matching(is_repo_readme)
    assert found.page_content == "opening"


def test_first_matching_returns_none_when_nothing_matches():
    index = BM25Index([impl_doc("code")])
    assert index.first_matching(is_repo_readme) is None


def test_readme_boost_prefers_the_opening_over_the_highest_scoring_chunk():
    # Reproduces the exact measured failure directly against the method
    # responsible for it, rather than fighting BM25 term-frequency arithmetic
    # to indirectly reconstruct crowding through the full pipeline.
    opening = readme_doc("This repository implements Hybrid-Retrieval RAG.", start=0)
    installation = readme_doc(
        "Clone the repository. Repository setup requires cloning the "
        "repository and installing repository dependencies for the repository.", start=900,
    )
    other = [impl_doc(f"unrelated helper {i}", start=i) for i in range(5)]
    index = BM25Index(other + [installation, opening])

    # Precondition: confirm search_within's own top pick really IS the wrong
    # chunk -- this is what makes the fix necessary, not incidental to it.
    naive_pick = index.search_within("what is the name of this repository", is_repo_readme, k=1)
    assert naive_pick and naive_pick[0][0].metadata["start_index"] == 900

    retriever = HybridRetriever(vector_retriever=StubVectorRetriever(other), bm25_index=index, final_k=5)
    fused_without_readme = list(other)  # simulates README being crowded out of the fused result
    result = retriever._surface_readme_if_missing(fused_without_readme, "what is the name of this repository")

    boosted = next((d for d in result if is_repo_readme(d)), None)
    assert boosted is not None, "README was not surfaced at all"
    assert boosted.metadata["start_index"] == 0, \
        f"boosted start_index={boosted.metadata['start_index']}, expected the opening (0)"
