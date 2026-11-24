"""
FROZEN copy of the retrieval pipeline that shipped before Phase 1+2.

This is the ``baseline_v3`` arm of the v3 benchmark: the exact retriever the
frozen baseline (results_phase0.json) measured - MMR dense search (k=10,
fetch_k=30, lambda 0.6), BM25 with the original tokenizer (top 10), equal-
weight RRF (k=60), final 5, README pin on. Copied verbatim from
src/hybrid_retrieval.py at commit dd87b3a so that later production changes
cannot move the baseline. Do not "fix" anything here: its value is that it
does not change. Only the Chroma store access goes through production code
(src/vector_store.load_vector_store), which this phase does not modify.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

BM25_AVAILABLE = True
FINAL_K = 5
VECTOR_CANDIDATE_K = 10
VECTOR_FETCH_K = 30
MMR_LAMBDA = 0.6
BM25_CANDIDATE_K = 10
RRF_SMOOTHING = 60
VECTOR_WEIGHT = 0.5
BM25_WEIGHT = 0.5

_CAMEL_LOWER_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_ACRONYM = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
is are was were be been being am do does did doing done
have has had having will would shall should can could may might must
i you he she it we they me him her us them my your his its our their
of in on at to for from by with without about into over under
as so such no not nor only own same too very just also
what which who whom whose when where why how
please tell explain give show
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    """Split identifier case boundaries, lowercase, split on alphanumeric runs,
    drop stopwords. Applied identically to corpus and query."""
    text = _CAMEL_ACRONYM.sub(" ", _CAMEL_LOWER_UPPER.sub(" ", text))
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]



# ── Chunk identity

def chunk_key(doc: Document) -> tuple:
    """
    Stable identity for a chunk, used to detect that both retrievers returned
    the same thing.

    ``start_index`` comes from ``add_start_index=True`` in chunk_documents, so
    (source, page, start_index) uniquely identifies a chunk for anything
    ingested by the current pipeline. Content hashing is the fallback for
    chunks indexed before that flag existed, and it also collapses genuinely
    identical text (repeated headers, boilerplate pages) which is desirable.
    """
    source = doc.metadata.get("source", "")
    start  = doc.metadata.get("start_index")
    if start is not None:
        return ("id", source, doc.metadata.get("page"), start)
    digest = hashlib.sha1(doc.page_content.encode("utf-8", "replace")).hexdigest()
    return ("hash", source, digest)


# ── Sparse index

class BM25Index:
    """
    Thin, deterministic wrapper over rank_bm25's Okapi BM25.

    Written directly against rank_bm25 rather than using
    ``langchain_community.retrievers.BM25Retriever`` for two reasons that
    matter here: this returns scores (so zero-score hits can be dropped before
    fusion), and an all-stopword query returns nothing instead of an arbitrary
    top-n slice of the corpus.
    """

    def __init__(self, documents: Sequence[Document]):
        # Chunks that tokenize to nothing (blank slides, OCR output that came
        # back as pure punctuation) are excluded rather than indexed empty.
        # Two reasons: rank_bm25 raises ZeroDivisionError when *every* document
        # is empty, and empty documents drag the average length down, which
        # distorts BM25's length normalisation for everything else. Nothing is
        # lost — a chunk with no tokens can never match a lexical query, and it
        # remains reachable through the vector index.
        indexed = [(d, tokenize(d.page_content)) for d in documents]
        indexed = [(d, toks) for d, toks in indexed if toks]

        self.documents: list[Document] = [d for d, _ in indexed]
        self._corpus_tokens: list[list[str]] = [toks for _, toks in indexed]
        self._token_sets: list[set[str]] = [set(toks) for _, toks in indexed]
        self.skipped_empty: int = len(documents) - len(indexed)

        self._bm25 = (
            BM25Okapi(self._corpus_tokens)
            if (BM25_AVAILABLE and self._corpus_tokens)
            else None
        )

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def ready(self) -> bool:
        return self._bm25 is not None

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """
        Top-k chunks that share at least one query term, ranked by BM25 score.

        Candidacy is decided by lexical overlap, NOT by a positive score.
        Okapi's IDF, log((N - n) + 0.5) - log(n + 0.5), goes negative for any
        term occurring in more than roughly half the corpus, and rank_bm25
        clamps those to a small negative epsilon. A chat holding one short
        upload is exactly that regime, so a `score > 0` filter would throw
        away genuine matches on small corpora.

        Restricting to overlapping chunks is still essential — without it a
        chunk sharing no term with the query would collect a rank, and
        therefore RRF mass, purely from its position in the corpus.
        """
        if not self.ready:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        wanted = set(query_tokens)
        candidates = [i for i, tokens in enumerate(self._token_sets) if tokens & wanted]
        if not candidates:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(candidates, key=lambda i: (-scores[i], i))[:k]  # index tiebreak = determinism
        return [(self.documents[i], float(scores[i])) for i in ranked]

    def search_within(self, query: str, predicate, k: int = 1) -> list[tuple[Document, float]]:
        """
        Best-scoring chunk(s) among documents matching ``predicate``, regardless
        of global rank.

        ``search()`` only considers documents that already rank in the overall
        top-k candidate pool. This method scores every document satisfying
        ``predicate`` directly, so a document that is objectively important
        (e.g. a repository's own README) can still be found even when raw term
        frequency in unrelated documents crowds it out of the normal pool
        entirely. See HybridRetriever._surface_readme_if_missing for why this
        exists: a module ABOUT handling repositories in general can use the
        word "repository" far more densely than a README describing one
        specific project, so BM25 legitimately, correctly by its own scoring
        rule, ranks the wrong file first.

        Same zero-relevance guard as search(): a query sharing no token with
        any matching document returns nothing rather than an arbitrary pick.
        """
        if not self.ready:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        candidates = [i for i, doc in enumerate(self.documents) if predicate(doc)]
        if not candidates:
            return []

        scores = self._bm25.get_scores(query_tokens)
        candidates = [i for i in candidates if scores[i] > 0.0]
        if not candidates:
            return []

        ranked = sorted(candidates, key=lambda i: (-scores[i], i))[:k]
        return [(self.documents[i], float(scores[i])) for i in ranked]

    def first_matching(self, predicate) -> Document | None:
        """
        The chunk with the lowest start_index among documents matching
        ``predicate`` — i.e. literally "the beginning of the file", not a
        BM25-ranked guess.

        Exists because a project's own identity (name, purpose) is almost
        always stated in a README's opening, while raw keyword density is
        not a reliable proxy for that. Measured case: for "what is the name
        of this repository", the highest-scoring README chunk under
        search_within() was the Installation section — "clone the
        repository" appears there — not the actual title, which mentions
        "repository" zero times. Term frequency and "states the project's
        identity" are different properties.
        """
        candidates = [d for d in self.documents if predicate(d)]
        if not candidates:
            return None
        return min(candidates, key=lambda d: d.metadata.get("start_index", 0))


# ── RRF

def reciprocal_rank_fusion(
    ranked_lists: dict[str, Iterable[Document]],
    weights: dict[str, float],
    smoothing: int = RRF_SMOOTHING,
    top_k: int = FINAL_K,
) -> list[Document]:
    """
    Merge several ranked Document lists into one.

    Returns fresh Document objects annotated with ``retrieval`` (which
    retrievers found the chunk) and ``rrf_score``. Fresh objects matter: the
    BM25 index caches its Documents for the whole session, so mutating their
    metadata in place would leak one query's provenance into the next.
    """
    scores:  dict[tuple, float] = {}
    origins: dict[tuple, list[str]] = {}
    docs:    dict[tuple, Document] = {}
    order:   list[tuple] = []

    for name, ranked in ranked_lists.items():
        weight = weights.get(name, 1.0)
        for rank, doc in enumerate(ranked, start=1):
            key = chunk_key(doc)
            if key not in docs:
                docs[key], scores[key], origins[key] = doc, 0.0, []
                order.append(key)
            scores[key] += weight / (smoothing + rank)
            if name not in origins[key]:
                origins[key].append(name)

    # Stable sort: ties keep first-seen order, so results are reproducible.
    fused = sorted(order, key=lambda key: -scores[key])[:top_k]

    return [
        Document(
            page_content=docs[key].page_content,
            metadata={
                **docs[key].metadata,
                "retrieval": "+".join(origins[key]),
                "rrf_score": round(scores[key], 6),
            },
        )
        for key in fused
    ]


def is_repo_readme(doc: Document) -> bool:
    """
    True for a repository's own top-level README — not a nested subdirectory
    readme (a submodule's readme is not "what is this project"), and not an
    uploaded document that happens to be named README.md (source_type check
    excludes that case).
    """
    return (
        doc.metadata.get("source_type") == "repo"
        and doc.metadata.get("source", "").lower() == "readme.md"
    )



def surface_readme_if_missing(bm25_index, fused: list[Document], query: str) -> list[Document]:
    """
    Guarantee a repository's top-level README a chance to be seen even when
    BM25's raw term-frequency scoring lets an unrelated implementation file
    crowd it out of the fused result entirely.

    Measured root cause (not a guess): for "what is this github repository
    about", the word "repository" appeared 57 times in repo_ingestion.py
    (a module ABOUT handling repositories as a general concept) versus 3
    times in that repo's own README.md (which describes THIS specific
    project) — so BM25 correctly, by its own scoring rule, ranked the wrong
    file first, and README did not even reach the BM25 candidate pool for
    a related follow-up question.

    Deliberately narrow: only activates when a repo README exists, is not
    already present, and has genuine positive lexical relevance to the
    query somewhere in it — it never forces in something with zero
    overlap. Relevance detection and chunk SELECTION are kept separate on
    purpose: search_within() answers "is README relevant at all", but the
    chunk it finds that way is whichever has the highest raw keyword
    density, which is frequently NOT where a project states its own
    identity (an Installation section saying "clone the repository"
    several times outscores a title that says it once). Once relevance is
    established, first_matching() supplies the actual chunk to inject —
    the file's opening, which is where identity information reliably
    lives regardless of the query's specific wording.

    Fixed as a targeted backstop rather than a global BM25 reweight,
    because the project's own retrieval evaluation already showed that
    reweighting BM25 globally can silently delete its genuine
    contributions elsewhere (see benchmark/TUNING.md, benchmark/RESULTS_V2.md).
    """
    if bm25_index is None or not bm25_index.ready:
        return fused
    if any(is_repo_readme(d) for d in fused):
        return fused  # README already earned its place naturally

    if not bm25_index.search_within(query, is_repo_readme, k=1):
        return fused  # no positive lexical relevance anywhere in it — do not force it in

    readme_doc = bm25_index.first_matching(is_repo_readme)
    if readme_doc is None:
        return fused

    boosted = Document(
        page_content=readme_doc.page_content,
        metadata={**readme_doc.metadata, "retrieval": "readme_boost"},
    )
    return fused[:-1] + [boosted]  # replace the weakest fused slot, not the strongest




class BaselineV3:
    """The old production retriever, assembled from the frozen pieces above."""

    def __init__(self, chroma_dir: str, final_k: int = FINAL_K):
        from src.vector_store import export_chunks, load_vector_store

        store = load_vector_store(chroma_dir)
        self.vector_retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": VECTOR_CANDIDATE_K, "fetch_k": VECTOR_FETCH_K,
                           "lambda_mult": MMR_LAMBDA},
        )
        self.bm25_index = BM25Index(export_chunks(chroma_dir))
        self.final_k = final_k

    def fused(self, query: str, top_k: int) -> list[Document]:
        dense = self.vector_retriever.invoke(query)
        sparse = [d for d, _ in self.bm25_index.search(query, BM25_CANDIDATE_K)]
        return reciprocal_rank_fusion({"vector": dense, "bm25": sparse},
                                      {"vector": VECTOR_WEIGHT, "bm25": BM25_WEIGHT},
                                      smoothing=RRF_SMOOTHING, top_k=top_k)

    def invoke(self, query: str, final_k: int | None = None) -> list[Document]:
        """Old HybridRetriever.invoke: fused top final_k, then the README pin."""
        return surface_readme_if_missing(self.bm25_index, self.fused(query, final_k or self.final_k), query)
