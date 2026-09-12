"""
Pre-registered configuration sweep over hypothetical retrieval settings.

NOTHING SHIPPED IS MODIFIED. Every configuration is assembled from building
blocks the project already exports:

  * ``HybridRetriever.final_k``  - an existing constructor field (C1, C2)
  * ``BM25Index.search``         - already returns (Document, score) pairs (C3)
  * ``reciprocal_rank_fusion``   - already a module-level function taking
                                   ranked lists, weights, smoothing and top_k

This measures hypothetical configurations. It does not adopt any of them.

A note on what final_k can and cannot do
----------------------------------------
``reciprocal_rank_fusion`` scores every candidate, sorts, and only then applies
``[:top_k]``. So final_k truncates but never reorders: Hit@1, Hit@3, Hit@5 and
MRR@5 are IDENTICAL for C0, C1 and C2 by construction, not by coincidence. What
final_k actually changes is how many chunks reach the prompt, so the
decision-relevant number for those configs is Hit@final_k, reported alongside.
"""

from __future__ import annotations

import os
import statistics
import sys

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document

from src.hybrid_retrieval import (
    BM25_CANDIDATE_K, BM25_WEIGHT, RRF_SMOOTHING, VECTOR_WEIGHT,
    reciprocal_rank_fusion,
)

# ── Pre-registered configurations ────────────────────────────────────────────
# Fixed before any of them was run, and not adjusted afterwards.

CONFIGS = {
    "C0_baseline": {
        "final_k": 5, "gate": False,
        "why": "the shipped configuration; reused from results.json, not recomputed",
    },
    "C1_k6": {
        "final_k": 6, "gate": False,
        "why": "minimal response to the 'relevant chunk sat at dense rank 4' finding",
    },
    "C2_k8": {
        "final_k": 8, "gate": False,
        "why": "tests whether the wider 13-query 'worse' group recovers too",
    },
    "W1_55_45": {
        "final_k": 5, "gate": False, "vector_weight": 0.55, "bm25_weight": 0.45,
        "why": ("v2: reweight RRF away from the equal-weight tie-break that "
                "produces strict interleaving. Pre-registered in this round; "
                "NOT previously published as a pre-registered config."),
    },
    "W2_60_40": {
        "final_k": 5, "gate": False, "vector_weight": 0.60, "bm25_weight": 0.40,
        "why": "v2: a stronger version of the same reweight hypothesis.",
    },
    "C3_gated": {
        "final_k": 5, "gate": True,
        "why": ("drop BM25 candidates below the median of that query's OWN "
                "candidate scores; self-normalising, no global constant, and "
                "specified without reference to which queries it fixes"),
    },
}


# ── C3's gate ────────────────────────────────────────────────────────────────

def bm25_median_gate(hits: list) -> list:
    """
    Keep BM25 candidates scoring at or above the median of this query's own
    candidate list.

    Self-normalising on purpose. BM25 scores are corpus- and query-relative
    with no absolute meaning, so any fixed cutoff would be a magic number
    tuned to one corpus. Taking the median of the query's own candidates needs
    no constant and cannot be reverse-engineered from a particular failure.

    Ties at the median are kept. With an even candidate count the median falls
    between two observations, so the rule keeps the upper half; with an odd
    count it keeps the upper half plus the median itself.

    `hits` is exactly what BM25Index.search returns: [(Document, score), ...].
    """
    if not hits:
        return []
    scores = [score for _doc, score in hits]
    cutoff = statistics.median(scores)
    return [doc for doc, score in hits if score >= cutoff]


# ── Retrieval under a configuration ──────────────────────────────────────────

def retrieve(config: dict, query: str, vector_retriever, bm25_index) -> list:
    """
    One configuration's ranked list for one query.

    Uses the same dense retriever object and the same BM25 index as the main
    evaluation, so the only thing varying between configs is the configuration.
    """
    dense = vector_retriever.invoke(query)

    sparse_hits = bm25_index.search(query, BM25_CANDIDATE_K) if bm25_index else []
    if config["gate"]:
        sparse = bm25_median_gate(sparse_hits)
    else:
        sparse = [doc for doc, _score in sparse_hits]

    return reciprocal_rank_fusion(
        {"vector": dense, "bm25": sparse},
        {"vector": config.get("vector_weight", VECTOR_WEIGHT),
         "bm25": config.get("bm25_weight", BM25_WEIGHT)},
        smoothing=RRF_SMOOTHING,
        top_k=config["final_k"],
    )


def gate_effect(query: str, bm25_index) -> dict:
    """How many BM25 candidates the gate removes for one query. Diagnostic."""
    hits = bm25_index.search(query, BM25_CANDIDATE_K)
    kept = bm25_median_gate(hits)
    return {"candidates": len(hits), "kept": len(kept),
            "dropped": len(hits) - len(kept)}
