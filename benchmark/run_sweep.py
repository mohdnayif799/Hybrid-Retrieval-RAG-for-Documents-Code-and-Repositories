#!/usr/bin/env python3
"""
Execute Part A (sign test on the existing harm tally) and Part B (the
pre-registered configuration sweep).

    python -m benchmark.run_sweep --store chroma_bench_<...>

No shipped file is modified and no API is called.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark import metrics
from benchmark.metrics import locator_of, normalise_locator
from benchmark.sweep import CONFIGS, gate_effect, retrieve
from evaluate_retrieval import (
    BUCKET_CODE, BUCKET_UNANSWERABLE, build_arms, load_queries,
)

RESULTS_JSON = os.path.join(BENCHMARK_DIR, "results.json")
SWEEP_JSON = os.path.join(BENCHMARK_DIR, "sweep_results.json")

NAMED_LOSSES = ("crs-03", "crs-06")


# ── Part A ───────────────────────────────────────────────────────────────────

def sign_test_on_harm(baseline: dict) -> dict:
    """
    Exact two-sided sign test on the non-tied rank changes.

    A supplement to the McNemar figure already reported, not a replacement.
    McNemar asks only whether Hit@5 flipped; this asks whether the rank moved
    at all, which is finer-grained and uses 21 observations instead of 3.
    """
    harm = baseline["harm"]
    worse, better = len(harm["worse"]), len(harm["better"])
    n = worse + better

    own = metrics.mcnemar_exact(worse, better)     # same exact binomial form

    scipy_p = None
    try:
        from scipy.stats import binomtest
        scipy_p = binomtest(min(worse, better), n, 0.5,
                            alternative="two-sided").pvalue
    except ImportError:
        pass

    return {
        "worse": worse, "better": better, "unchanged": harm["unchanged"],
        "n_non_tied": n,
        "p_value": own,
        "scipy_binomtest_p": scipy_p,
        "agrees_with_scipy": (scipy_p is not None
                              and abs(own - min(1.0, scipy_p)) < 1e-12),
        "significant_at_05": own < 0.05,
    }


# ── Part B ───────────────────────────────────────────────────────────────────

def run_config(name: str, config: dict, queries: list,
               vector_retriever, bm25_index) -> dict:
    """Per-query ranked locators and top-1 RRF score under one configuration."""
    out = {"locators": {}, "top1_score": {}, "provenance": {}}
    for q in queries:
        docs = retrieve(config, q["question"], vector_retriever, bm25_index)
        out["locators"][q["id"]] = [locator_of(d) for d in docs]
        out["top1_score"][q["id"]] = (
            docs[0].metadata.get("rrf_score") if docs else None)
        out["provenance"][q["id"]] = [d.metadata.get("retrieval") for d in docs]
    return out


def bucket1_metrics(locators_by_id: dict, queries: list, final_k: int) -> dict:
    """Hit@1/3/5, MRR@5, plus Hit@final_k - the decision-relevant cutoff."""
    ids = [q["id"] for q in queries if q["category"] in BUCKET_CODE]
    rel = {q["id"]: {normalise_locator(e) for e in q["relevant"]}
           for q in queries if q["id"] in ids}

    def prop(k):
        hits = sum(metrics.hit_at_k(locators_by_id[i], rel[i], k) for i in ids)
        return metrics.proportion(hits, len(ids))

    return {
        "n": len(ids),
        "hit@1": prop(1),
        "hit@3": prop(3),
        "hit@5": prop(5),
        f"hit@final_k({final_k})": prop(final_k),
        "mrr@5": sum(metrics.mrr_at_k(locators_by_id[i], rel[i], 5)
                     for i in ids) / len(ids),
        "ranks": {i: metrics.first_relevant_rank(locators_by_id[i], rel[i])
                  for i in ids},
    }


def recovery_check(ranks: dict, final_k: int) -> dict:
    """Are the two named total losses inside what the generator would see?"""
    out = {}
    for qid in NAMED_LOSSES:
        rank = ranks.get(qid)
        out[qid] = {"rank": rank,
                    "recovered": rank is not None and rank <= final_k}
    return out


def harm_vs_baseline(baseline_ranks: dict, config_ranks: dict,
                     queries: list) -> dict:
    """
    Rank change per query against C0, so a fix that repairs the two named
    losses while breaking something else is visible rather than hidden.
    """
    MISS = 999
    cat = {q["id"]: q["category"] for q in queries}
    better, worse, unchanged = [], [], 0
    for qid, base in baseline_ranks.items():
        new = config_ranks.get(qid)
        b_eff = base if base is not None else MISS
        n_eff = new if new is not None else MISS
        entry = {"id": qid, "category": cat.get(qid),
                 "c0_rank": base, "config_rank": new}
        if n_eff < b_eff:
            better.append(entry)
        elif n_eff > b_eff:
            worse.append(entry)
        else:
            unchanged += 1
    return {"better": better, "worse": worse, "unchanged": unchanged,
            "n_better": len(better), "n_worse": len(worse)}


def bucket3_rate(top1: dict, queries: list) -> dict:
    """
    Above-threshold rate on the unanswerable set, threshold re-derived per
    configuration from that configuration's own answerable scores.
    """
    una = [q["id"] for q in queries if q["category"] in BUCKET_UNANSWERABLE]
    ans = [q["id"] for q in queries if q["category"] not in BUCKET_UNANSWERABLE]

    ans_scores = sorted(s for s in (top1[i] for i in ans) if s is not None)
    threshold = ans_scores[len(ans_scores) // 2] if ans_scores else None

    above = sum(1 for i in una
                if top1[i] is not None and threshold is not None
                and top1[i] >= threshold)
    returned = sum(1 for i in una if top1[i] is not None)
    return {
        "threshold": threshold,
        "threshold_basis": f"median top-1 score over {len(ans_scores)} answerable queries",
        "above_threshold": metrics.proportion(above, len(una)),
        "returned_any": metrics.proportion(returned, len(una)),
        "per_query": {i: top1[i] for i in una},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    args = parser.parse_args()

    with open(RESULTS_JSON, encoding="utf-8") as handle:
        baseline = json.load(handle)

    queries = load_queries()
    arms, ctx = build_arms(args.store)
    vector_retriever = ctx["vector_retriever"]
    bm25_index = ctx["bm25_index"]

    report = {"store": os.path.basename(args.store),
              "configs": {k: v for k, v in CONFIGS.items()}}

    # ── Part A ──────────────────────────────────────────────────────────────
    report["part_a_sign_test"] = sign_test_on_harm(baseline)

    # ── Part B ──────────────────────────────────────────────────────────────
    # C0 as published, for continuity with RESULTS.md.
    c0_published = {qid: row["hybrid"]["rank"]
                    for qid, row in baseline["per_query"].items()
                    if row["_meta"]["category"] in BUCKET_CODE}

    # C0 recomputed IN THIS PROCESS. This is the valid harm baseline.
    #
    # Chroma's ANN behaviour is stable within a process but can differ between
    # processes on the same store directory - merely opening and querying a
    # store rewrites chroma.sqlite3. Comparing a config measured now against a
    # baseline measured in an earlier process therefore attributes a retrieval
    # wobble to the configuration change. Recomputing C0 here means every
    # config sees the identical dense candidate list, so any difference is the
    # configuration and nothing else.
    c0_run = run_config("C0_inprocess", CONFIGS["C0_baseline"], queries,
                        vector_retriever, bm25_index)
    c0_inproc = bucket1_metrics(c0_run["locators"], queries, 5)
    c0_ranks = c0_inproc["ranks"]

    report["c0_reproducibility"] = {
        "note": ("C0 recomputed in-process vs the published run; differences "
                 "are cross-process ANN variation, not configuration effects"),
        "mismatches": {
            qid: {"published": c0_published[qid], "in_process": c0_ranks[qid]}
            for qid in c0_ranks if c0_published.get(qid) != c0_ranks[qid]
        },
    }
    report["c0_inprocess_bucket1"] = {
        k: v for k, v in c0_inproc.items() if k != "ranks"
    }
    report["c0_inprocess_bucket3"] = bucket3_rate(c0_run["top1_score"], queries)

    per_config = {}
    for name, config in CONFIGS.items():
        if name == "C0_baseline":
            b1 = {
                "n": baseline["bucket1_code"]["n"],
                "hit@1": baseline["bucket1_code"]["arms"]["hybrid"]["hit@1"],
                "hit@3": baseline["bucket1_code"]["arms"]["hybrid"]["hit@3"],
                "hit@5": baseline["bucket1_code"]["arms"]["hybrid"]["hit@5"],
                "hit@final_k(5)": baseline["bucket1_code"]["arms"]["hybrid"]["hit@5"],
                "mrr@5": baseline["bucket1_code"]["arms"]["hybrid"]["mrr@5"],
                "ranks": c0_ranks,
            }
            b3 = {
                "threshold": baseline["bucket3_unanswerable"]["arms"]["hybrid"]["threshold"],
                "above_threshold": baseline["bucket3_unanswerable"]["arms"]["hybrid"]["above_threshold"],
                "returned_any": baseline["bucket3_unanswerable"]["arms"]["hybrid"]["returned_any_chunk"],
                "per_query": {r["id"]: r["score"] for r in
                              baseline["bucket3_unanswerable"]["arms"]["hybrid"]["per_query"]},
                "threshold_basis": "reused from results.json",
            }
            per_config[name] = {
                "config": config, "bucket1": b1,
                "recovery": recovery_check(c0_ranks, config["final_k"]),
                "harm_vs_c0": {"better": [], "worse": [], "unchanged": len(c0_ranks),
                               "n_better": 0, "n_worse": 0},
                "bucket3": b3, "reused": True,
            }
            continue

        run = run_config(name, config, queries, vector_retriever, bm25_index)
        b1 = bucket1_metrics(run["locators"], queries, config["final_k"])
        per_config[name] = {
            "config": config,
            "bucket1": b1,
            "recovery": recovery_check(b1["ranks"], config["final_k"]),
            "harm_vs_c0": harm_vs_baseline(c0_ranks, b1["ranks"], queries),
            "bucket3": bucket3_rate(run["top1_score"], queries),
            "reused": False,
        }

    # Gate diagnostics: how much C3 actually removes.
    effects = [gate_effect(q["question"], bm25_index) for q in queries]
    with_cands = [e for e in effects if e["candidates"] > 0]
    report["gate_diagnostics"] = {
        "queries_with_bm25_candidates": len(with_cands),
        "total_candidates": sum(e["candidates"] for e in with_cands),
        "total_dropped": sum(e["dropped"] for e in with_cands),
        "drop_distribution": dict(Counter(e["dropped"] for e in with_cands)),
    }

    report["per_config"] = per_config
    with open(SWEEP_JSON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"written: {SWEEP_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
