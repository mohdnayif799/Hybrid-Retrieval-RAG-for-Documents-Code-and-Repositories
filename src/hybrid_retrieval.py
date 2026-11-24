"""
Hybrid retrieval: a recall-oriented candidate pool, reordered by a reranker.

Two stages
----------
1. Candidate pool (this module). Plain dense similarity (top ``POOL_K``)
   and BM25 (top ``POOL_K``) over the same chunks, fused by min-max score
   blending with equal weight. The pool's job is recall: get the right chunk
   into the top ``RERANK_TOP_K``, not into the top 5.
2. Reranking (src/reranker.py). A cross-encoder reads each (question, chunk)
   pair of the pool's top ``RERANK_TOP_K`` and the best ``FINAL_K`` go into
   the prompt.

Why this shape, with the measured numbers (v3 benchmark, dev split), is in
docs/DECISIONS.md. In short: MMR was removed (LangChain's Chroma wrapper
returned its selection in plain similarity order, so it only dropped
candidates), the pool went from 20 MMR+BM25 candidates to a 50+50 blend
(dev Recall@20 84.2 -> 89.5), and the reranker raised dev Hit@1 from 39.8 to
54.9 and Hit@5 from 73.5 to 81.4.

The two retrievers fail in opposite directions, which is why both feed the
pool:

  * Dense  - matches meaning. Handles paraphrase ("how do I cancel" ~
    "termination of the agreement"), but a 384-dim MiniLM vector smears rare
    tokens: identifiers, section numbers, acronyms.
  * BM25   - matches surface form. Nails "Section 4.2", "CVE-2024-3094" and
    pasted identifiers, but is blind to synonyms.

Fallbacks: retrieval never breaks
---------------------------------
  * BM25 unavailable (rank_bm25 missing, index build fails) -> dense-only pool.
  * Reranker unavailable, raising, or over its time budget -> fused order.
  * Fewer chunks than POOL_K (a small chat) -> the pool is whatever exists.

Each query writes one log line (pool size, rerank time, top-1 score, fallback)
to stderr. The question text and chunk content are never logged.

Design constraints inherited from the existing app
--------------------------------------------------
1. Exactly one live Chroma client per directory. Opening a second one on the
   same SQLite file locks and loses data on Windows (see the root-cause note
   in ``vector_store.append_to_vector_store``). The BM25 corpus is therefore
   read back through the *cached* client via ``vector_store.export_chunks``,
   never through a fresh ``Chroma(...)``.
2. ``build_rag_chain`` runs on every uncached message, so index construction
   must be cached. ``load_bm25_index`` is keyed on (chroma_dir, chunk_count);
   appending documents changes the count, which invalidates the entry for
   free - no manual cache busting needed. The reranker is loaded once per
   process (src/reranker.get_reranker).
3. Returned Documents keep ``source`` / ``page`` / ``start_index`` untouched
   so the citation helpers keep working unchanged.
4. Streamlit and Chroma are imported lazily / defensively so this module can
   be imported by pytest with neither installed.
"""

from __future__ import annotations

import functools
import hashlib
import re
import sys
import threading
import time
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.reranker import RERANK_BUDGET_S, RERANK_TOP_K, RerankFailed

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised only on a broken install
    BM25Okapi = None
    BM25_AVAILABLE = False


# ── Tuning constants ──────────────────────────────────────────────────────────
# Measured on the v3 benchmark's dev split; see docs/DECISIONS.md.

FINAL_K      = 5      # chunks that go into the prompt (unchanged context budget)
POOL_K       = 50     # candidates each retriever contributes to the pool
FUSION_ALPHA = 0.5    # weight of the dense side in the min-max blend
# RERANK_TOP_K (20) and RERANK_BUDGET_S (3 s) live in src/reranker.py.

# Legacy constants of the pre-v3 pipeline (MMR + RRF). Not used by the default
# path; kept because the v1/v2 benchmark tooling (evaluate_retrieval.py,
# benchmark/sweep.py) imports them to reproduce its own measurements.
VECTOR_CANDIDATE_K = 10
VECTOR_FETCH_K     = 30
MMR_LAMBDA         = 0.6
BM25_CANDIDATE_K   = 10
RRF_SMOOTHING      = 60
VECTOR_WEIGHT      = 0.5
BM25_WEIGHT        = 0.5


# ── Tokenisation ──────────────────────────────────────────────────────────────
# Identical for corpus and query - consistency between the two sides matters
# far more than linguistic sophistication.
#
# Identifiers are kept WHOLE as well as split into their parts:
#
#   "TooManyRedirects"          -> toomanyredirects, too, many, redirects
#   "get_encoding_from_headers" -> get_encoding_from_headers, get, encoding,
#                                  from, headers
#
# Two consequences, both measured on the v3 benchmark:
#   * an exact paste of an identifier matches the identifier itself, not just
#     a scatter of common words ("get", "auth", "url") that occur everywhere;
#   * stopwords are removed only OUTSIDE identifiers. The old tokenizer turned
#     "TooManyRedirects" into ["many", "redirects"] because "too" is a
#     stopword, and lost that query entirely.
# A lowercased identifier typed as prose ("toomanyredirects") now matches the
# whole-identifier token too.
#
# A word counts as an identifier when it contains an underscore or an internal
# case boundary (camelCase, PascalCase, HTTPServer). Plain words are
# lowercased, split on non-alphanumerics and stopword-filtered as before, so
# "GPT-4" still indexes as ["gpt", "4"].

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAMEL_LOWER_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")   # buildHybrid -> build Hybrid
_CAMEL_ACRONYM     = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])") # HTTPServer  -> HTTP Server

# Function words carry almost no discriminative signal and inflate document
# length, which BM25 penalises. Removed from both sides, but only outside
# identifiers (see above).
#
# NOTE ON PROGRAMMING KEYWORDS: this list also contains `if`, `for`, `in`,
# `is`, `not`, `and`, `or`, `as`, `from`, `with` and `this`, which are keywords
# in most languages, so they are unsearchable as bare terms. That is a
# deliberate decision, not an oversight. BM25Index.search() uses token overlap
# to decide *candidacy*, so admitting a token present in nearly every code
# chunk would make nearly every chunk a candidate for any query containing it,
# while IDF gives those terms almost no ranking power in return. "for loop"
# still matches on "loop". Meaningful code keywords - `class`, `def`,
# `function`, `return`, `import`, `async`, `await`, `try`, `except`, `self` -
# are absent from this list and remain searchable.
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


def _is_identifier(word: str) -> bool:
    core = word.strip("_")
    if "_" in core:
        return True
    return bool(_CAMEL_LOWER_UPPER.search(core) or _CAMEL_ACRONYM.search(core))


def tokenize(text: str) -> list[str]:
    """Identifiers: whole token plus all parts. Other words: lowercase,
    alphanumeric runs, stopwords dropped. Applied identically to corpus and query."""
    out: list[str] = []
    for word in _WORD_RE.findall(text):
        if _is_identifier(word):
            whole = word.strip("_").lower()
            out.append(whole)
            split = _CAMEL_ACRONYM.sub(" ", _CAMEL_LOWER_UPPER.sub(" ", word)).lower()
            out.extend(p for p in _TOKEN_RE.findall(split) if p != whole)
        else:
            out.extend(t for t in _TOKEN_RE.findall(word.lower()) if t not in _STOPWORDS)
    return out


# ── Chunk identity ────────────────────────────────────────────────────────────

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


# ── Sparse index ──────────────────────────────────────────────────────────────

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


# ── Fusion ────────────────────────────────────────────────────────────────────
# reciprocal_rank_fusion is the pre-v3 fusion; the default path now uses
# minmax_fusion below. Kept for the v1/v2 benchmark tooling.

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


def _minmax(scores: dict) -> dict:
    """Scale one retriever's scores to [0, 1]; a single or flat list maps to 1.0."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    return {k: (v - lo) / (hi - lo) if hi > lo else 1.0 for k, v in scores.items()}


def minmax_fusion(
    dense: Sequence[tuple[Document, float]],
    sparse: Sequence[tuple[Document, float]],
    alpha: float = FUSION_ALPHA,
) -> list[Document]:
    """
    Blend two scored lists: alpha * minmax(dense) + (1 - alpha) * minmax(bm25).

    A chunk missing from one list contributes 0 from that side. Ties keep
    first-seen order (dense first), so a fixed candidate set always fuses to
    the same order. Returns fresh Documents carrying ``retrieval`` (which
    retrievers found the chunk) and ``fusion_score``; the BM25 index caches its
    Documents for the session, so they are never mutated in place.

    Score blending was chosen over RRF for the candidate pool because, measured
    on the v3 dev split at depth 50, it put more gold chunks into the top 20
    (Recall@20 89.5 vs 86.4) - the reranker's input window. Per-query min-max
    is unstable when a list is very short; here each side contributes up to 50
    candidates, and a one-item list simply scores 1.0.
    """
    docs: dict[tuple, Document] = {}
    origins: dict[tuple, list[str]] = {}
    dense_scores: dict[tuple, float] = {}
    sparse_scores: dict[tuple, float] = {}
    order: list[tuple] = []
    for name, ranked, sink in (("vector", dense, dense_scores), ("bm25", sparse, sparse_scores)):
        for doc, score in ranked:
            key = chunk_key(doc)
            if key not in docs:
                docs[key], origins[key] = doc, []
                order.append(key)
            if key not in sink:
                sink[key] = float(score)
                origins[key].append(name)
    dn, sn = _minmax(dense_scores), _minmax(sparse_scores)
    fused = {k: alpha * dn.get(k, 0.0) + (1 - alpha) * sn.get(k, 0.0) for k in order}
    ranked_keys = sorted(order, key=lambda k: -fused[k])     # stable: first-seen on ties
    return [
        Document(
            page_content=docs[k].page_content,
            metadata={**docs[k].metadata, "retrieval": "+".join(origins[k]),
                      "fusion_score": round(fused[k], 6)},
        )
        for k in ranked_keys
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



# ── Retriever ─────────────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Candidate pool (dense + BM25, min-max fused) -> cross-encoder rerank -> top FINAL_K.

    Keeps the ``.invoke(query) -> list[Document]`` contract of the retriever it
    replaced, so rag_chain's call site is unchanged.

    The dense side is ``vector_store`` (a Chroma store, queried with scores) or,
    for tests and adapters, any ``vector_retriever`` exposing ``invoke`` that
    returns a ranked list; rank then stands in for score.
    """

    vector_store: Any = None
    vector_retriever: Any = None
    bm25_index: Any = None
    reranker: Any = None
    corpus_size: int | None = None
    final_k: int = FINAL_K
    pool_k: int = POOL_K
    rerank_k: int = RERANK_TOP_K
    fusion_alpha: float = FUSION_ALPHA
    rerank_budget_s: float = RERANK_BUDGET_S
    # The README pin is off by default: on the v3 dev split it never turned a
    # repository-identity miss into a hit and, after reranking, cost two other
    # queries their only gold chunk (docs/DECISIONS.md). Kept as an option.
    surface_readme: bool = False
    log_stream: Any = None

    # ── stage 1: candidate pool ──────────────────────────────────────────────

    def _dense(self, query: str) -> list[tuple[Document, float]]:
        k = self.pool_k if self.corpus_size is None else min(self.pool_k, self.corpus_size)
        if k <= 0:
            return []
        if self.vector_store is not None:
            # Chroma returns L2 distances; with normalised embeddings that is a
            # linear function of cosine similarity, so -distance preserves both
            # the order and the min-max-normalised value of cosine.
            hits = self.vector_store.similarity_search_with_score(query, k=k)
            return [(doc, -float(dist)) for doc, dist in hits]
        if self.vector_retriever is not None:
            docs = list(self.vector_retriever.invoke(query))[:k]
            n = len(docs)
            return [(doc, float(n - i)) for i, doc in enumerate(docs)]
        return []

    def _sparse(self, query: str) -> list[tuple[Document, float]]:
        if self.bm25_index is None or not getattr(self.bm25_index, "ready", False):
            return []
        return self.bm25_index.search(query, self.pool_k)

    def candidate_pool(self, query: str) -> list[Document]:
        """The fused pool, best first, before reranking (up to 2 x pool_k chunks)."""
        return minmax_fusion(self._dense(query), self._sparse(query), self.fusion_alpha)

    # ── stage 2: rerank ──────────────────────────────────────────────────────

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        pool = self.candidate_pool(query)
        candidates = pool[: self.rerank_k]
        results = pool[: self.final_k]
        fallback, rerank_ms, top1 = "none", None, None

        if not candidates:
            fallback = "empty_pool"
        elif self.reranker is None:
            fallback = "no_reranker"
        else:
            started = time.monotonic()
            try:
                scores = self.reranker.score(query, [d.page_content for d in candidates],
                                             budget_s=self.rerank_budget_s)
                order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
                results = [
                    Document(page_content=candidates[i].page_content,
                             metadata={**candidates[i].metadata,
                                       "rerank_score": round(scores[i], 4)})
                    for i in order[: self.final_k]
                ]
                top1 = scores[order[0]]
            except RerankFailed as exc:
                fallback = exc.reason                 # "timeout" | "error": keep fused order
            rerank_ms = round(1000 * (time.monotonic() - started))

        if self.surface_readme:
            results = self._surface_readme_if_missing(results, query)

        log_retrieval(pool_size=len(pool), reranked=len(candidates) if rerank_ms is not None else 0,
                      rerank_ms=rerank_ms, top1_score=top1, fallback=fallback,
                      bm25=self.bm25_index is not None and getattr(self.bm25_index, "ready", False),
                      stream=self.log_stream)
        return results

    def _surface_readme_if_missing(self, fused: list[Document], query: str) -> list[Document]:
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
        if self.bm25_index is None or not self.bm25_index.ready:
            return fused
        if any(is_repo_readme(d) for d in fused):
            return fused  # README already earned its place naturally

        if not self.bm25_index.search_within(query, is_repo_readme, k=1):
            return fused  # no positive lexical relevance anywhere in it — do not force it in

        readme_doc = self.bm25_index.first_matching(is_repo_readme)
        if readme_doc is None:
            return fused

        boosted = Document(
            page_content=readme_doc.page_content,
            metadata={**readme_doc.metadata, "retrieval": "readme_boost"},
        )
        return fused[:-1] + [boosted]  # replace the weakest fused slot, not the strongest


# ── Caching shim ──────────────────────────────────────────────────────────────
# Mirrors st.cache_resource's semantics (including .clear()) so this module is
# importable and testable without a Streamlit runtime.

try:
    import streamlit as st
    _cache_resource = st.cache_resource
except ModuleNotFoundError:  # pragma: no cover - pytest / CLI path
    def _cache_resource(**_kwargs):
        def _decorator(fn):
            return functools.lru_cache(maxsize=8)(fn)
        return _decorator


@_cache_resource(show_spinner="Building lexical (BM25) index…")
def load_bm25_index(chroma_dir: str, chunk_count: int) -> BM25Index:
    """
    Build (once) the BM25 index for a chat's chunks.

    ``chunk_count`` is not used in the body — it is part of the cache key.
    Appending documents changes the count and therefore invalidates this
    entry automatically, which keeps the sparse index in lockstep with the
    vector store without any explicit cache management.
    """
    from src.vector_store import export_chunks  # lazy: keeps Chroma out of test imports

    chunks = export_chunks(chroma_dir)
    index = BM25Index(chunks)
    print(f"[INFO] BM25 index built over {len(index)} chunk(s) from '{chroma_dir}'.")
    return index


# ── Releasing cached indexes ─────────────────────────────────────────────────
# The (chroma_dir, chunk_count) key invalidates correctly on append, but the
# superseded entry used to stay cached for the life of the process: one
# stale index per append, per chat. This records the count each directory's
# index was last built for, so the previous entry can be dropped when the
# count changes, and a store's entry can be dropped when it is evicted.

_BM25_BUILT_FOR: dict[str, int] = {}
_BM25_LOCK = threading.Lock()


def _clear_bm25_entry(chroma_dir: str, chunk_count: int) -> None:
    clear = getattr(load_bm25_index, "clear", None)
    if clear is not None:
        clear(chroma_dir, chunk_count)      # st.cache_resource: this entry only
    else:                                   # pragma: no cover - lru_cache shim
        load_bm25_index.cache_clear()


def _record_bm25_count(chroma_dir: str, chunk_count: int) -> None:
    """Remember the count about to be used; drop the entry it supersedes."""
    with _BM25_LOCK:
        previous = _BM25_BUILT_FOR.get(chroma_dir)
        _BM25_BUILT_FOR[chroma_dir] = chunk_count
    if previous is not None and previous != chunk_count:
        _clear_bm25_entry(chroma_dir, previous)


def forget_bm25_index(chroma_dir: str) -> None:
    """Drop the cached BM25 index of a store that is being evicted."""
    with _BM25_LOCK:
        chunk_count = _BM25_BUILT_FOR.pop(chroma_dir, None)
    if chunk_count is not None:
        _clear_bm25_entry(chroma_dir, chunk_count)


def build_hybrid_retriever(chroma_dir: str) -> HybridRetriever:
    """
    Assemble the retriever used by the RAG pipeline.

    Never fails for a missing component: without rank_bm25 or a buildable BM25
    index the pool is dense-only, and without a reranker the pool's fused
    order is used. Each degradation prints a warning.
    """
    from src.reranker import get_reranker
    from src.vector_store import count_chunks, load_vector_store

    store = load_vector_store(chroma_dir)   # cached client; never a second connection
    chunk_count = count_chunks(chroma_dir)

    bm25_index = None
    if not BM25_AVAILABLE:
        print("[WARNING] rank_bm25 not installed - candidate pool is dense-only. "
              "Run: pip install rank-bm25")
    else:
        try:
            _record_bm25_count(chroma_dir, chunk_count)
            bm25_index = load_bm25_index(chroma_dir, chunk_count)
        except Exception as exc:
            # Retrieval quality should degrade, not the app.
            print(f"[WARNING] BM25 index unavailable ({exc}) - candidate pool is dense-only.")
            bm25_index = None

    return HybridRetriever(vector_store=store, bm25_index=bm25_index,
                           reranker=get_reranker(), corpus_size=chunk_count)


# ── Logging ───────────────────────────────────────────────────────────────────

def format_retrieval_log(pool_size: int, reranked: int, rerank_ms: int | None,
                         top1_score: float | None, fallback: str, bm25: bool) -> str:
    """
    One line per retrieval. Takes only numbers and fixed labels: the question
    and the chunk text are not parameters, so they cannot leak into logs.
    fallback: none | no_reranker | timeout | error | empty_pool.
    """
    ms = "-" if rerank_ms is None else str(rerank_ms)
    top1 = "-" if top1_score is None else f"{top1_score:.3f}"
    return (f"retrieval pool={pool_size} reranked={reranked} rerank_ms={ms} "
            f"top1_score={top1} fallback={fallback} bm25={'on' if bm25 else 'off'}")


def log_retrieval(pool_size: int, reranked: int, rerank_ms: int | None,
                  top1_score: float | None, fallback: str, bm25: bool, stream=None) -> None:
    print(format_retrieval_log(pool_size, reranked, rerank_ms, top1_score, fallback, bm25),
          file=stream or sys.stderr, flush=True)
