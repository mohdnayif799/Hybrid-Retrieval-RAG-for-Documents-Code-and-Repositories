"""
Cross-encoder reranking of the hybrid candidate pool.

Retrieval is two-stage. ``hybrid_retrieval`` builds a candidate pool (dense +
BM25) tuned for recall; this module reorders the pool's top
``RERANK_TOP_K`` by reading each (question, chunk) pair jointly, and the
best ``FINAL_K`` go into the prompt. Why, with the numbers, is in
docs/DECISIONS.md.

Failure never breaks retrieval:
  * the model cannot be loaded (missing files, ``RAG_RERANKER=off``)
    -> ``get_reranker()`` returns None and the caller keeps fused order;
  * scoring raises, or exceeds ``RERANK_BUDGET_S`` -> ``RerankFailed`` and the
    caller keeps fused order for that query.

The budget is checked between batches of ``RERANK_BATCH`` pairs rather than
by interrupting a running model call, so a slow query overshoots by at most
one batch (~0.25 s on one vCPU) and no thread is left computing in the
background. Small batches were also measured to be faster and lighter than
large ones on CPU (less padding): 20 pairs at batch 4 took p50 1.04 s /
p95 1.28 s on one thread and added 138 MB of peak memory.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_MAX_LENGTH = 512
RERANK_BATCH = 4
RERANK_TOP_K = 20
RERANK_BUDGET_S = 3.0
# Not used for refusal yet (see benchmark/v3/README.md, open items): on the
# v3 dev split, a top-1 score below about -7.5 kept ~95% of answerable
# questions and flagged 6 of 14 unanswerable ones.
REFUSAL_SCORE_HINT = -7.5

_ENV_SWITCH = "RAG_RERANKER"         # "off" disables reranking entirely


class RerankFailed(RuntimeError):
    """Scoring failed or ran out of budget; the caller keeps fused order."""

    def __init__(self, reason: str, message: str = ""):
        super().__init__(message or reason)
        self.reason = reason            # "timeout" | "error"


class Reranker:
    """A loaded cross-encoder with batched, time-budgeted scoring."""

    def __init__(self, model: Any, batch_size: int = RERANK_BATCH):
        self.model = model
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str], budget_s: float = RERANK_BUDGET_S,
              clock=time.monotonic) -> list[float]:
        """One relevance score per text, or ``RerankFailed``."""
        start = clock()
        scores: list[float] = []
        try:
            for i in range(0, len(texts), self.batch_size):
                pairs = [(query, t) for t in texts[i:i + self.batch_size]]
                scores.extend(float(s) for s in self.model.predict(pairs, batch_size=self.batch_size))
                if clock() - start > budget_s and i + self.batch_size < len(texts):
                    raise RerankFailed("timeout", f"over the {budget_s:.1f}s budget")
        except RerankFailed:
            raise
        except Exception as exc:  # any model failure means "keep fused order"
            raise RerankFailed("error", type(exc).__name__) from exc
        if clock() - start > budget_s:
            raise RerankFailed("timeout", f"over the {budget_s:.1f}s budget")
        return scores


_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"loaded": False, "reranker": None, "load_seconds": None}


def enabled() -> bool:
    return os.environ.get(_ENV_SWITCH, "on").strip().lower() not in {"off", "0", "false", "no"}


def get_reranker() -> Reranker | None:
    """
    The process-wide reranker, loaded on first call (once, under a lock, so a
    question arriving during the startup warm-up waits for that load instead
    of starting a second one). None if disabled or if loading fails.
    """
    if _STATE["loaded"]:
        return _STATE["reranker"]
    with _LOCK:
        if _STATE["loaded"]:
            return _STATE["reranker"]
        reranker = None
        started = time.monotonic()
        if enabled():
            try:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(RERANK_MODEL, device="cpu", max_length=RERANK_MAX_LENGTH)
                reranker = Reranker(model)
            except Exception as exc:  # noqa: BLE001 - degrade to fused order
                print(f"[WARNING] reranker unavailable ({type(exc).__name__}); "
                      "retrieval keeps fused order.", file=sys.stderr, flush=True)
        _STATE.update(loaded=True, reranker=reranker,
                      load_seconds=round(time.monotonic() - started, 2))
        if reranker is not None:
            print(f"[INFO] reranker {RERANK_MODEL} loaded in {_STATE['load_seconds']}s",
                  file=sys.stderr, flush=True)
        return reranker


def reset_for_tests() -> None:
    with _LOCK:
        _STATE.update(loaded=False, reranker=None, load_seconds=None)


def set_reranker_for_tests(reranker: Reranker | None) -> None:
    with _LOCK:
        _STATE.update(loaded=True, reranker=reranker, load_seconds=0.0)


_WARMUP_STARTED = threading.Event()


def start_background_warmup() -> bool:
    """
    Load the embedding model and the reranker in a daemon thread, once per
    process. Called when the app first renders, so the first question does not
    pay for model loading. Returns True if this call started the warm-up.
    Disabled with RAG_WARMUP=0 (the test suite does this).
    """
    if os.environ.get("RAG_WARMUP", "1").strip() in {"0", "off", "false", "no"}:
        return False
    if _WARMUP_STARTED.is_set():
        return False
    _WARMUP_STARTED.set()

    def _warm() -> None:
        try:
            from src.vector_store import get_embeddings

            get_embeddings().embed_query("warm-up")
        except Exception as exc:  # noqa: BLE001 - warm-up is best effort
            print(f"[WARNING] embedding warm-up failed ({type(exc).__name__})", file=sys.stderr)
        reranker = get_reranker()
        if reranker is not None:
            try:
                reranker.score("warm-up", ["warm-up"], budget_s=60.0)
            except RerankFailed:
                pass

    threading.Thread(target=_warm, name="model-warmup", daemon=True).start()
    return True
