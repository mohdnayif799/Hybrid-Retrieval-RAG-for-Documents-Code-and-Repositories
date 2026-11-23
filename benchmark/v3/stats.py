"""
Paired percentile bootstrap for per-query retrieval metrics.

Every arm is scored on the same queries, so differences are estimated PAIRED:
one resample of query indices is drawn and applied to every arm, and the
difference of means is taken inside each resample. That keeps the
between-query variance (some questions are simply harder) out of the
difference, which is where most of the width of an unpaired interval comes
from.

Fixed seed and a fixed resample matrix per group of queries, so a report is
reproducible bit for bit.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261123
ALPHA = 0.05


def resample_matrix(n: int, resamples: int = DEFAULT_RESAMPLES,
                    seed: int = DEFAULT_SEED) -> np.ndarray:
    """(resamples, n) matrix of query indices drawn with replacement."""
    if n <= 0:
        raise ValueError("cannot bootstrap an empty set of queries")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(resamples, n))


def _interval(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mean_ci(values, idx: np.ndarray, alpha: float = ALPHA) -> dict:
    """Mean of per-query values with a percentile bootstrap interval."""
    v = np.asarray(values, dtype=float)
    if v.shape[0] != idx.shape[1]:
        raise ValueError("values and resample matrix disagree on the number of queries")
    lo, hi = _interval(v[idx].mean(axis=1), alpha)
    return {"mean": float(v.mean()), "ci_low": lo, "ci_high": hi, "n": int(v.shape[0])}


def paired_diff_ci(a, b, idx: np.ndarray, alpha: float = ALPHA) -> dict:
    """mean(a) - mean(b), paired over the same queries, with its interval."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arms must be scored on the same queries")
    d = a - b
    draws = d[idx].mean(axis=1)
    lo, hi = _interval(draws, alpha)
    return {"diff": float(d.mean()), "ci_low": lo, "ci_high": hi, "n": int(d.shape[0]),
            "excludes_zero": bool(lo > 0 or hi < 0)}
