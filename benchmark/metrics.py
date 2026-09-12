"""
Pure metric functions for the retrieval benchmark.

Deliberately free of Chroma, Streamlit and the dataset: everything here takes
plain lists and returns numbers, so it can be tested against hand-computed
fixtures. The harness does the I/O; this module does the arithmetic.
"""

from __future__ import annotations

import math
import os

# ── Locator identity ─────────────────────────────────────────────────────────
# Ground truth is file/page level, never chunk level, so it survives a change
# to chunk size. These two functions are the only place that rule is encoded.


def locator_of(chunk) -> tuple:
    """The (file, page) identity of a retrieved chunk."""
    source = chunk.metadata.get("source", "")
    if chunk.metadata.get("source_type") == "repo":
        return (source, None)
    return (os.path.basename(source), chunk.metadata.get("page"))


def normalise_locator(entry) -> tuple:
    """A YAML locator -> the same tuple shape as locator_of()."""
    if isinstance(entry, str):
        return (entry, None)
    if isinstance(entry, dict) and "file" in entry:
        return (entry["file"], entry.get("page"))
    raise ValueError(f"unrecognised locator: {entry!r}")


# ── Ground-truth specifications ──────────────────────────────────────────────
# Two forms, because (file, page) cannot address a location inside a .docx:
# Docx2txtLoader returns ONE Document at page 0, so all 99 chunks of the ETCE
# document share a single (file, page) locator. A file-level hit there would be
# satisfied by any chunk of a 78,000-character document, which measures
# nothing. Range specs address a character span instead, which is what
# add_start_index=True already records.

def normalise_spec(entry) -> tuple:
    """
    A YAML ground-truth entry -> a matching specification.

      "src/requests/utils.py"                  -> ("exact", path, None)
      {file: "notes.pdf", page: 1}             -> ("exact", "notes.pdf", 1)
      {file: "unit.docx", start: 40296, end: 43036}
                                               -> ("range", "unit.docx", a, b)
    """
    if isinstance(entry, str):
        return ("exact", entry, None)
    if isinstance(entry, dict) and "file" in entry:
        if "start" in entry or "end" in entry:
            start = int(entry.get("start", 0))
            end = int(entry["end"]) if entry.get("end") is not None else start
            if end < start:
                raise ValueError(f"range end < start in {entry!r}")
            return ("range", entry["file"], start, end)
        return ("exact", entry["file"], entry.get("page"))
    raise ValueError(f"unrecognised ground-truth entry: {entry!r}")


def _chunk_file(chunk) -> str:
    source = chunk.metadata.get("source", "")
    if chunk.metadata.get("source_type") == "repo":
        return source
    return os.path.basename(source)


def chunk_matches(spec: tuple, chunk) -> bool:
    """Does this retrieved chunk satisfy this ground-truth specification?"""
    kind = spec[0]
    if kind == "exact":
        return locator_of(chunk) == (spec[1], spec[2])
    if kind == "range":
        _, file, start, end = spec
        if _chunk_file(chunk) != file:
            return False
        begin = chunk.metadata.get("start_index")
        if begin is None:
            return False
        finish = begin + len(chunk.page_content)
        # Half-open interval overlap. A chunk counts if it carries any part of
        # the annotated span - the ASCII table in the ETCE document spans three
        # chunks, so requiring containment would make it unhittable.
        return begin < end and start < finish
    raise ValueError(f"unknown spec kind: {kind!r}")


# ── Generic rank core ────────────────────────────────────────────────────────

def _first_match_rank(items, predicate) -> int | None:
    for rank, item in enumerate(items, start=1):
        if predicate(item):
            return rank
    return None


def first_relevant_chunk_rank(chunks, specs) -> int | None:
    """1-based rank of the first chunk satisfying any specification."""
    return _first_match_rank(
        chunks, lambda c: any(chunk_matches(s, c) for s in specs))


def hit_at_k_chunks(chunks, specs, k: int) -> bool:
    if not specs:
        raise ValueError("hit_at_k_chunks is undefined for an empty spec set")
    return first_relevant_chunk_rank(chunks[:k], specs) is not None


def mrr_at_k_chunks(chunks, specs, k: int = 5) -> float:
    if not specs:
        raise ValueError("mrr_at_k_chunks is undefined for an empty spec set")
    rank = first_relevant_chunk_rank(chunks[:k], specs)
    return 1.0 / rank if rank else 0.0


def recall_at_k_chunks(chunks, specs, k: int = 5) -> float:
    """Share of the DISTINCT specifications satisfied within the top k."""
    if not specs:
        raise ValueError("recall_at_k_chunks is undefined for an empty spec set")
    window = chunks[:k]
    found = sum(1 for s in specs if any(chunk_matches(s, c) for c in window))
    return found / len(specs)


# ── Rank-based metrics ───────────────────────────────────────────────────────

def first_relevant_rank(retrieved_locators: list, relevant: set) -> int | None:
    """1-based rank of the first relevant item, or None if absent."""
    for rank, loc in enumerate(retrieved_locators, start=1):
        if loc in relevant:
            return rank
    return None


def hit_at_k(retrieved_locators: list, relevant: set, k: int) -> bool:
    """
    True if any of the top k is relevant.

    UNDEFINED for an empty relevant set - callers must not ask. Hit@K on a
    query with no correct answer is not 0, it is meaningless, and averaging a
    fabricated 0 into a mean silently drags every aggregate down.
    """
    if not relevant:
        raise ValueError("hit_at_k is undefined for an empty relevant set")
    rank = first_relevant_rank(retrieved_locators[:k], relevant)
    return rank is not None


def mrr_at_k(retrieved_locators: list, relevant: set, k: int = 5) -> float:
    """Reciprocal rank of the first relevant item within k, else 0.0."""
    if not relevant:
        raise ValueError("mrr_at_k is undefined for an empty relevant set")
    rank = first_relevant_rank(retrieved_locators[:k], relevant)
    return 1.0 / rank if rank else 0.0


def recall_at_k(retrieved_locators: list, relevant: set, k: int = 5) -> float:
    """
    Share of the distinct relevant locators present in the top k.

    Only meaningful when a query has several genuinely required locators,
    which is why the harness reports it for the cross_file subset alone.
    Elsewhere it collapses to Hit@K and would just be a second name for it.
    """
    if not relevant:
        raise ValueError("recall_at_k is undefined for an empty relevant set")
    found = {loc for loc in retrieved_locators[:k] if loc in relevant}
    return len(found) / len(relevant)


# ── Confidence intervals ─────────────────────────────────────────────────────

def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple:
    """
    Wilson score interval for a binomial proportion (default 95%).

    Wilson rather than the normal approximation because at n=35 with
    proportions near 0 or 1 the normal interval runs outside [0, 1] and is
    simply wrong. Returns (low, high); (0.0, 0.0) for n == 0 so callers can
    render an empty cell instead of dividing by zero.
    """
    if n == 0:
        return (0.0, 0.0)
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} out of range for n={n}")

    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    low, high = max(0.0, centre - margin), min(1.0, centre + margin)

    # At p = 0 the Wilson lower bound is exactly 0, and at p = 1 the upper
    # bound is exactly 1. Floating point lands a few 1e-17 off, which breaks
    # the defining property that the interval contains the point estimate.
    # Pin the exact endpoints rather than let a rounding artefact through.
    if successes == 0:
        low = 0.0
    if successes == n:
        high = 1.0
    return (low, high)


# ── Paired significance ──────────────────────────────────────────────────────

def mcnemar_exact(b: int, c: int) -> float:
    """
    Two-sided exact McNemar test on the discordant pairs only.

    b = A succeeded and B failed; c = A failed and B succeeded. Concordant
    pairs carry no information about a difference and are excluded by design.

    The exact binomial form is used rather than the chi-square approximation
    because the discordant count here is small (single digits), which is
    exactly where the approximation misbehaves. Returns 1.0 when b == c == 0:
    with no discordant pairs there is no evidence of any difference.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0

    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def discordant_pairs(a_success: list, b_success: list) -> tuple:
    """
    (b, c) discordant counts for two arms evaluated on the same queries.

    b = a won, b lost. c = b won, a lost.
    """
    if len(a_success) != len(b_success):
        raise ValueError("arms must be scored on the same queries")
    b = sum(1 for x, y in zip(a_success, b_success) if x and not y)
    c = sum(1 for x, y in zip(a_success, b_success) if y and not x)
    return b, c


# ── Aggregation helper ───────────────────────────────────────────────────────

def proportion(successes: int, n: int) -> dict:
    """A proportion with its Wilson interval, ready for a report table."""
    low, high = wilson_ci(successes, n)
    return {
        "successes": successes,
        "n": n,
        "value": (successes / n) if n else None,
        "ci_low": low,
        "ci_high": high,
    }
