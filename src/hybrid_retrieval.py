"""
Hybrid retrieval: dense (Chroma + MMR) fused with sparse (BM25) via RRF.

Why hybrid
----------
The two retrievers fail in opposite directions, which is exactly what makes
them worth combining:

  * Dense / MMR  — matches meaning. Handles paraphrase well
    ("how do I cancel" ~ "termination of the agreement"), but a 384-dim
    MiniLM vector smears rare tokens. Exact identifiers, section numbers,
    acronyms, product names and typo-prone proper nouns often do not survive
    the projection.
  * Sparse / BM25 — matches surface form. Nails "Section 4.2", "ISO 27001",
    "CVE-2024-3094", any term the user copied verbatim out of the document,
    but is blind to synonyms and returns nothing when vocabulary differs.

Fusion strategy
---------------
Reciprocal Rank Fusion (Cormack et al., 2009) rather than score blending.
Chroma returns cosine distances, rank_bm25 returns unbounded corpus-relative
scores; the two have no common scale, and per-query min-max normalisation is
unstable when one list is short or its scores are near-uniform. RRF ignores
magnitudes entirely and combines *ranks*:

    score(chunk) = sum over retrievers of  weight / (smoothing + rank)

A chunk found by both retrievers accumulates two terms and therefore beats a
chunk found by only one, which is the behaviour we want.

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
   free — no manual cache busting needed.
3. Returned Documents keep ``source`` / ``page`` / ``start_index`` untouched
   so the citation helpers in app.py keep working unchanged.
4. Streamlit and Chroma are imported lazily / defensively so this module can
   be imported by pytest or a future evaluate.py with neither installed.
"""

from __future__ import annotations

import functools
import hashlib
import re
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised only on a broken install
    BM25Okapi = None
    BM25_AVAILABLE = False


# ── Tuning constants ──────────────────────────────────────────────────────────
# Each retriever proposes CANDIDATE_K chunks; RRF fuses both lists and the top
# FINAL_K survive into the prompt. FINAL_K stays at 5 so the QA prompt keeps
# roughly the same context budget as before the upgrade.

FINAL_K            = 5
VECTOR_CANDIDATE_K = 10    # dense candidates fed into fusion
VECTOR_FETCH_K     = 30    # MMR pre-fetch pool (was 20 for k=5)
MMR_LAMBDA         = 0.6   # unchanged: 0 = max diversity, 1 = pure relevance
BM25_CANDIDATE_K   = 10    # sparse candidates fed into fusion

RRF_SMOOTHING = 60         # standard constant; damps the top-rank advantage
VECTOR_WEIGHT = 0.5
BM25_WEIGHT   = 0.5


# ── Tokenisation ──────────────────────────────────────────────────────────────
# Deliberately simple and identical for corpus and query — consistency between
# the two sides matters far more than linguistic sophistication. Lowercasing
# plus alphanumeric splitting means "GPT-4" indexes as ["gpt", "4"] and the
# query "gpt 4" still matches it.

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Identifier case boundaries, applied BEFORE lowercasing.
#
# Without this, casing is destroyed before it can be used as a word boundary,
# and the result is language-dependent by accident: `_` is not in [a-z0-9] so
# snake_case splits correctly, while camelCase and PascalCase collapse into
# one token. `build_hybrid_retriever` was reachable from the query "build
# hybrid retriever"; `buildHybridRetriever` was not, so BM25 returned nothing
# for every natural-language query against a JS/TS/Java/Go codebase — and
# because zero lexical overlap means an empty candidate list, hybrid retrieval
# silently degraded to vector-only on exactly the identifier queries where
# dense embeddings are weakest.
#
# Splitting both corpus and query keeps the module's core invariant (identical
# treatment on both sides), so an exact paste of `buildHybridRetriever` still
# matches — it now decomposes the same way the document did.
_CAMEL_LOWER_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")   # buildHybrid -> build Hybrid
_CAMEL_ACRONYM     = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])") # HTTPServer  -> HTTP Server

# Function words carry almost no discriminative signal and inflate document
# length, which BM25 penalises. Removed from both sides so the length
# normalisation stays meaningful.
#
# NOTE ON PROGRAMMING KEYWORDS: this list also contains `if`, `for`, `in`,
# `is`, `not`, `and`, `or`, `as`, `from`, `with` and `this`, which are keywords
# in most languages, so they are unsearchable as bare terms. That is a
# deliberate decision, not an oversight. Stopwords here are not merely a
# scoring tweak — BM25Index.search() uses token overlap to decide *candidacy*,
# so admitting a token present in nearly every code chunk would make nearly
# every chunk a candidate for any query containing it, while IDF gives those
# terms almost no ranking power in return. The realistic cost is queries made
# only of keywords ("for", "if"), which are not useful searches; "for loop"
# still matches on "loop". Meaningful code keywords — `class`, `def`,
# `function`, `return`, `import`, `async`, `await`, `try`, `except`, `self` —
# are all absent from this list and remain searchable.
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


def tokenize(text: str) -> list[str]:
    """Split identifier case boundaries, lowercase, split on alphanumeric runs,
    drop stopwords. Applied identically to corpus and query."""
    text = _CAMEL_ACRONYM.sub(" ", _CAMEL_LOWER_UPPER.sub(" ", text))
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


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


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

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


# ── Retriever ─────────────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Drop-in replacement for the previous ``vector_store.as_retriever(...)``.

    Exposes the same ``.invoke(query) -> list[Document]`` contract, so
    rag_chain's call site is unchanged and the retriever stays composable with
    the rest of LangChain.
    """

    vector_retriever: Any
    bm25_index: Any = None
    final_k: int = FINAL_K
    bm25_candidate_k: int = BM25_CANDIDATE_K
    vector_weight: float = VECTOR_WEIGHT
    bm25_weight: float = BM25_WEIGHT
    rrf_smoothing: int = RRF_SMOOTHING

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        dense = self.vector_retriever.invoke(query)

        sparse: list[Document] = []
        if self.bm25_index is not None and self.bm25_index.ready:
            sparse = [d for d, _ in self.bm25_index.search(query, self.bm25_candidate_k)]

        # No lexical hits (unknown vocabulary, or an all-stopword query):
        # fusion of a single list is order-preserving, so this degrades
        # cleanly to the original dense-only behaviour.
        return reciprocal_rank_fusion(
            {"vector": dense, "bm25": sparse},
            {"vector": self.vector_weight, "bm25": self.bm25_weight},
            smoothing=self.rrf_smoothing,
            top_k=self.final_k,
        )


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


def build_hybrid_retriever(chroma_dir: str):
    """
    Assemble the retriever used by the RAG pipeline.

    Falls back to dense-only retrieval — with a warning rather than an
    exception — if rank_bm25 is missing, so a partial install degrades the
    answer quality instead of taking the app down.
    """
    from src.vector_store import count_chunks, load_vector_store

    store = load_vector_store(chroma_dir)   # cached client; never a second connection
    vector_retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           VECTOR_CANDIDATE_K,
            "fetch_k":     VECTOR_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )

    if not BM25_AVAILABLE:
        print("[WARNING] rank_bm25 not installed — falling back to vector-only "
              "retrieval. Run: pip install rank-bm25")
        return vector_retriever

    try:
        bm25_index = load_bm25_index(chroma_dir, count_chunks(chroma_dir))
    except Exception as exc:
        # Retrieval quality should degrade, not the app. Vector search alone
        # is still a working system.
        print(f"[WARNING] BM25 index unavailable ({exc}) — using vector-only retrieval.")
        return vector_retriever

    return HybridRetriever(vector_retriever=vector_retriever, bm25_index=bm25_index)
