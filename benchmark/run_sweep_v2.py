#!/usr/bin/env python3
"""
v2 configuration validation on the expanded query set.

Tests a PRE-EXISTING hypothesis on new data. This is not a search for a
favourable configuration: the configs are fixed in benchmark/sweep.py before
this runs, and every one of them is reported whichever way it lands.

Adds two arms the earlier round lacked:
  * dense_only at final_k=8, so hybrid@8 is compared at EQUAL budget rather
    than against dense@5;
  * provenance per configuration, to test whether reweighting eliminates
    BM25-only finds rather than merely re-ranking them.

    python -m benchmark.run_sweep_v2 --store chroma_bench_<...>
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
from benchmark.sweep import CONFIGS, retrieve
from evaluate_retrieval import (
    BUCKET_CODE, BUCKET_DOC, BUCKET_UNANSWERABLE, build_arms, load_queries,
)

OUT = os.path.join(BENCHMARK_DIR, "sweep_v2_results.json")
NAMED_LOSSES = ("crs-03", "crs-06")


def summarise(docs_by_id: dict, queries: list, categories: tuple,
              final_k: int) -> dict:
    ids = [q["id"] for q in queries if q["category"] in categories]
    specs = {q["id"]: [metrics.normalise_spec(e) for e in q["relevant"]]
             for q in queries if q["id"] in ids}

    def prop(k):
        hits = sum(metrics.hit_at_k_chunks(docs_by_id[i], specs[i], k) for i in ids)
        return metrics.proportion(hits, len(ids))

    ranks = {i: metrics.first_relevant_chunk_rank(docs_by_id[i], specs[i])
             for i in ids}
    return {
        "n": len(ids),
        "hit@1": prop(1), "hit@3": prop(3), "hit@5": prop(5),
        f"hit@final_k({final_k})": prop(final_k),
        "mrr@5": sum(metrics.mrr_at_k_chunks(docs_by_id[i], specs[i], 5)
                     for i in ids) / len(ids),
        "ranks": ranks,
    }


def provenance(docs_by_id: dict, queries: list, categories: tuple) -> dict:
    """Which retriever surfaced the first relevant chunk, per config."""
    specs = {q["id"]: [metrics.normalise_spec(e) for e in q["relevant"]]
             for q in queries if q["category"] in categories}
    counts = Counter()
    for qid, sp in specs.items():
        rank = metrics.first_relevant_chunk_rank(docs_by_id[qid], sp)
        if rank is None:
            continue
        counts[docs_by_id[qid][rank - 1].metadata.get("retrieval") or "?"] += 1
    return dict(counts)


def bucket3(docs_by_id: dict, queries: list) -> dict:
    una = [q["id"] for q in queries if q["category"] in BUCKET_UNANSWERABLE]
    ans = [q["id"] for q in queries if q["category"] not in BUCKET_UNANSWERABLE]

    def top1(qid):
        d = docs_by_id[qid]
        return d[0].metadata.get("rrf_score") if d else None

    scores = sorted(s for s in (top1(i) for i in ans) if s is not None)
    threshold = scores[len(scores) // 2] if scores else None
    above = sum(1 for i in una
                if top1(i) is not None and threshold is not None
                and top1(i) >= threshold)
    return {"threshold": threshold,
            "above_threshold": metrics.proportion(above, len(una))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    args = parser.parse_args()

    queries = load_queries()
    arms, ctx = build_arms(args.store)
    vr, bi = ctx["vector_retriever"], ctx["bm25_index"]

    report = {"store": os.path.basename(args.store), "per_config": {}}

    # Hybrid configurations.
    for name, config in CONFIGS.items():
        docs = {q["id"]: retrieve(config, q["question"], vr, bi) for q in queries}
        fk = config["final_k"]
        report["per_config"][name] = {
            "config": {k: v for k, v in config.items() if k != "why"},
            "why": config["why"],
            "bucket1": summarise(docs, queries, BUCKET_CODE, fk),
            "bucket2": summarise(docs, queries, BUCKET_DOC, fk),
            "provenance_b1": provenance(docs, queries, BUCKET_CODE),
            "provenance_b2": provenance(docs, queries, BUCKET_DOC),
            "bucket3": bucket3(docs, queries),
            "recovery": {
                qid: {"rank": summarise(docs, queries, BUCKET_CODE, fk)["ranks"].get(qid),
                      "recovered": (summarise(docs, queries, BUCKET_CODE, fk)
                                    ["ranks"].get(qid) or 10 ** 6) <= fk}
                for qid in NAMED_LOSSES
            },
        }

    # Dense-only arms. D8 closes the unequal-budget gap: the earlier round only
    # ever compared hybrid@8 against dense@5.
    for label, k in (("D0_dense_k5", 5), ("D1_dense_k8", 8)):
        docs = {q["id"]: vr.invoke(q["question"])[:k] for q in queries}
        report["per_config"][label] = {
            "config": {"final_k": k, "arm": "dense_only"},
            "why": ("dense-only at the SAME budget as the hybrid config it is "
                    "compared against"),
            "bucket1": summarise(docs, queries, BUCKET_CODE, k),
            "bucket2": summarise(docs, queries, BUCKET_DOC, k),
            "provenance_b1": {}, "provenance_b2": {},
            "bucket3": {"threshold": None, "above_threshold": None},
            "recovery": {
                qid: {"rank": summarise(docs, queries, BUCKET_CODE, k)["ranks"].get(qid),
                      "recovered": (summarise(docs, queries, BUCKET_CODE, k)
                                    ["ranks"].get(qid) or 10 ** 6) <= k}
                for qid in NAMED_LOSSES
            },
        }

    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
