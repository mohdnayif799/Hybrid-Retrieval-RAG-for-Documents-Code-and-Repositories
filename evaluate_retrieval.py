#!/usr/bin/env python3
"""
Retrieval-only evaluation harness. API-FREE by construction.

Kept separate from any answer-quality evaluation on purpose: this measures
retrieval, which happens entirely before generation, so it needs no model
call, no API key and none of the rate-limit pacing an LLM-judge script needs.

Three arms, one store, one query set:

  dense_only  the MMR retriever's ranked list, truncated to FINAL_K
  bm25_only   BM25Index.search at FINAL_K over the same chunks
  hybrid      the shipped HybridRetriever (MMR + BM25 fused with RRF)

    python evaluate_retrieval.py --determinism   # build twice, compare
    python evaluate_retrieval.py                 # full evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark import metrics
from benchmark.metrics import locator_of, normalise_locator
from src.hybrid_retrieval import (
    BM25_CANDIDATE_K, FINAL_K, MMR_LAMBDA, RRF_SMOOTHING, VECTOR_CANDIDATE_K,
    VECTOR_FETCH_K, BM25Index, HybridRetriever, chunk_key,
)

RESULTS_JSON = os.path.join(PROJECT_ROOT, "benchmark", "results.json")

# Buckets. The corpus is 1111 code chunks against 12 document chunks, so a
# document query competes with a pool ~90x smaller than a code query does.
# Averaging those two regimes into one number would report a ceiling effect
# and a real effect as though they were the same measurement.
BUCKET_CODE = ("exact_identifier", "paraphrase", "conceptual",
               "literal_string", "cross_file")
# v2: the document bucket gained three sub-types. doc_discrimination is the
# hard one - its answers sit in clusters of near-duplicate, similarly-worded
# sections, so retrieving a sibling is a failure. Tagged separately so that
# capability is reported on its own rather than blended into the easier
# factual/conceptual categories.
BUCKET_DOC = ("doc_factual", "doc_conceptual", "doc_paraphrase",
              "doc_exact_term", "doc_discrimination")
BUCKET_UNANSWERABLE = ("unanswerable",)


# ══════════════════════════════════════════════════════════════════════════
# Arms
# ══════════════════════════════════════════════════════════════════════════

def build_arms(chroma_dir: str):
    """
    Construct all three arms from ONE shared MMR retriever.

    The dense-only arm is not a re-parameterised second retriever; it is the
    same object the hybrid consumes, truncated. MMR selects greedily from a
    fetch_k pool, so the first FINAL_K of a k=VECTOR_CANDIDATE_K selection are
    exactly the k=FINAL_K selection *provided fetch_k is unchanged*. Sharing
    the object makes fetch_k and lambda_mult identical by construction rather
    than by two call sites that agree today and drift tomorrow.
    """
    from src.vector_store import count_chunks, export_chunks, load_vector_store

    store = load_vector_store(chroma_dir)

    # The single dense retriever. Identical to build_hybrid_retriever's.
    vector_retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           VECTOR_CANDIDATE_K,   # 10
            "fetch_k":     VECTOR_FETCH_K,       # 30
            "lambda_mult": MMR_LAMBDA,           # 0.6
        },
    )

    chunks = export_chunks(chroma_dir)
    bm25_index = BM25Index(chunks)

    hybrid = HybridRetriever(
        vector_retriever=vector_retriever,      # the SAME object
        bm25_index=bm25_index,
        final_k=FINAL_K,
    )

    def dense_only(query: str):
        # Truncation of the shared fetch_k=30 selection, not a second call.
        return vector_retriever.invoke(query)[:FINAL_K]

    def bm25_only(query: str):
        return [doc for doc, _score in bm25_index.search(query, FINAL_K)]

    def hybrid_arm(query: str):
        return hybrid.invoke(query)

    return {
        "dense_only": dense_only,
        "bm25_only": bm25_only,
        "hybrid": hybrid_arm,
    }, {"store": store, "vector_retriever": vector_retriever,
        "bm25_index": bm25_index, "n_chunks": count_chunks(chroma_dir),
        "chunks": chunks}


def fresh_store(tag: str) -> str:
    """Build a brand-new benchmark store, clearing every cache first."""
    from benchmark.build_corpus import build_store
    from src import vector_store
    from src.hybrid_retrieval import load_bm25_index

    # Without this the cached client and BM25 index from a previous build
    # would be reused and "determinism" would be measuring nothing.
    vector_store.load_vector_store.clear()
    load_bm25_index.clear()

    started = time.time()
    chroma_dir, chunks, _meta = build_store(progress=False)
    print(f"  [{tag}] built {len(chunks)} chunks -> "
          f"{os.path.basename(chroma_dir)} in {time.time()-started:.1f}s")
    return chroma_dir


# ══════════════════════════════════════════════════════════════════════════
# Running queries
# ══════════════════════════════════════════════════════════════════════════

def load_queries():
    from benchmark.validate_dataset import load_dataset
    return load_dataset()["queries"]


def run_arms(arms: dict, queries: list) -> dict:
    """{arm: {query_id: [chunk_key, ...]}} - full ranked lists."""
    out: dict = {name: {} for name in arms}
    for name, fn in arms.items():
        for q in queries:
            docs = fn(q["question"])
            out[name][q["id"]] = [chunk_key(d) for d in docs]
    return out


# ══════════════════════════════════════════════════════════════════════════
# Step A.1 - determinism gate
# ══════════════════════════════════════════════════════════════════════════

def determinism_check() -> dict:
    """
    Build the store twice from scratch and compare every ranked list.

    Chroma assigns fresh internal UUIDs on each build, so identity is compared
    on chunk_key (source, page, start_index) - the same identity RRF uses to
    decide two retrievers found the same chunk. Comparing UUIDs would report
    a spurious difference on every single query.
    """
    queries = load_queries()
    print(f"determinism: {len(queries)} queries x 3 arms x 2 independent builds\n")

    dir_a = fresh_store("build A")
    arms_a, meta_a = build_arms(dir_a)
    results_a = run_arms(arms_a, queries)

    dir_b = fresh_store("build B")
    arms_b, meta_b = build_arms(dir_b)
    results_b = run_arms(arms_b, queries)

    report = {"store_a": os.path.basename(dir_a), "store_b": os.path.basename(dir_b),
              "n_chunks_a": meta_a["n_chunks"], "n_chunks_b": meta_b["n_chunks"],
              "arms": {}}

    for arm in results_a:
        identical, differing = 0, []
        for q in queries:
            qid = q["id"]
            a, b = results_a[arm][qid], results_b[arm][qid]
            if a == b:
                identical += 1
            else:
                same_set = set(a) == set(b)
                differing.append({
                    "id": qid,
                    "category": q["category"],
                    "kind": "reordered (same chunks)" if same_set
                            else "different chunks",
                    "top1_same": bool(a and b and a[0] == b[0]),
                    "a": [f"{k[1]}@{k[3]}" for k in a],
                    "b": [f"{k[1]}@{k[3]}" for k in b],
                })
        report["arms"][arm] = {
            "identical": identical,
            "total": len(queries),
            "differing": differing,
        }

    return report


# ══════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════

def score_answerable(arms: dict, queries: list, ctx: dict) -> dict:
    """
    Per-query scoring for every query that HAS a ground truth.

    Unanswerable queries are excluded here and handled separately: Hit@K is
    undefined, not zero, when no correct answer exists.
    """
    per_query: dict = defaultdict(dict)

    for q in queries:
        if q["category"] in BUCKET_UNANSWERABLE:
            continue
        # Specs, not bare locators: the .docx added in v2 has a single
        # (file, page) identity across all 99 of its chunks, so its ground
        # truth is expressed as start_index ranges instead.
        specs = [metrics.normalise_spec(e) for e in q["relevant"]]

        for arm_name, fn in arms.items():
            docs = fn(q["question"])
            locs = [locator_of(d) for d in docs]
            rank = metrics.first_relevant_chunk_rank(docs, specs)
            per_query[q["id"]][arm_name] = {
                "locators": locs,
                "rank": rank,
                "hit@1": metrics.hit_at_k_chunks(docs, specs, 1),
                "hit@3": metrics.hit_at_k_chunks(docs, specs, 3),
                "hit@5": metrics.hit_at_k_chunks(docs, specs, 5),
                "mrr@5": metrics.mrr_at_k_chunks(docs, specs, 5),
                "recall@5": metrics.recall_at_k_chunks(docs, specs, 5),
                "provenance": [d.metadata.get("retrieval") for d in docs],
                "span_hit": span_hit(q, docs),
            }
        per_query[q["id"]]["_meta"] = {"category": q["category"],
                                      "question": q["question"]}
    return per_query


def span_hit(q: dict, docs: list) -> bool | None:
    """
    Did any retrieved chunk overlap the annotated span?

    None when the query carries no span, so the span-level metric is reported
    over its own denominator rather than counting unannotated queries as
    misses. The gap between file-level and span-level Hit@5 is the chunking
    diagnostic: right file, wrong part of it.
    """
    span = q.get("_span")
    if not span:
        return None
    target_loc, start, end = span["locator"], span["start"], span["end"]
    for d in docs:
        if locator_of(d) != target_loc:
            continue
        d_start = d.metadata.get("start_index")
        if d_start is None:
            continue
        d_end = d_start + len(d.page_content)
        if d_start < end and start < d_end:      # interval overlap
            return True
    return False


def bucket_summary(per_query: dict, categories: tuple, arms: list) -> dict:
    """Aggregate one bucket, every proportion carrying a Wilson interval."""
    ids = [qid for qid, row in per_query.items()
           if row["_meta"]["category"] in categories]
    summary: dict = {"n": len(ids), "arms": {}}

    for arm in arms:
        rows = [per_query[qid][arm] for qid in ids]
        summary["arms"][arm] = {
            "hit@1": metrics.proportion(sum(r["hit@1"] for r in rows), len(rows)),
            "hit@3": metrics.proportion(sum(r["hit@3"] for r in rows), len(rows)),
            "hit@5": metrics.proportion(sum(r["hit@5"] for r in rows), len(rows)),
            "mrr@5": (sum(r["mrr@5"] for r in rows) / len(rows)) if rows else None,
        }
        # Span-level Hit@5 over its own denominator.
        spans = [r["span_hit"] for r in rows if r["span_hit"] is not None]
        summary["arms"][arm]["span_hit@5"] = (
            metrics.proportion(sum(spans), len(spans)) if spans else None
        )

    # Per-category breakdown - reported before the aggregate.
    summary["by_category"] = {}
    for cat in categories:
        cat_ids = [qid for qid in ids if per_query[qid]["_meta"]["category"] == cat]
        summary["by_category"][cat] = {
            "n": len(cat_ids),
            "arms": {
                arm: metrics.proportion(
                    sum(per_query[qid][arm]["hit@5"] for qid in cat_ids), len(cat_ids)
                ) for arm in arms
            },
        }

    # Recall@5 on cross_file only, where several locators are genuinely needed.
    cf_ids = [qid for qid in ids
              if per_query[qid]["_meta"]["category"] == "cross_file"]
    if cf_ids:
        summary["cross_file_recall@5"] = {
            arm: sum(per_query[qid][arm]["recall@5"] for qid in cf_ids) / len(cf_ids)
            for arm in arms
        }

    # Paired dense-vs-hybrid comparison, discordant pairs reported explicitly.
    if "dense_only" in arms and "hybrid" in arms and ids:
        dense = [per_query[qid]["dense_only"]["hit@5"] for qid in ids]
        hybrid = [per_query[qid]["hybrid"]["hit@5"] for qid in ids]
        b, c = metrics.discordant_pairs(dense, hybrid)
        summary["mcnemar"] = {
            "dense_won": b, "hybrid_won": c, "discordant": b + c,
            "p_value": metrics.mcnemar_exact(b, c),
            "underpowered": (b + c) < 6,
        }
    return summary


# ══════════════════════════════════════════════════════════════════════════
# Provenance and harm
# ══════════════════════════════════════════════════════════════════════════

def provenance_attribution(per_query: dict, categories: tuple) -> dict:
    """
    For each hybrid success, which retriever surfaced the FIRST relevant chunk?

    A mechanistic claim about when BM25 earns its place, which is stronger
    evidence than an aggregate delta.
    """
    overall = Counter()
    by_category: dict = defaultdict(Counter)
    for qid, row in per_query.items():
        cat = row["_meta"]["category"]
        if cat not in categories:
            continue
        h = row["hybrid"]
        if not h["hit@5"] or h["rank"] is None:
            continue
        origin = h["provenance"][h["rank"] - 1] or "unknown"
        overall[origin] += 1
        by_category[cat][origin] += 1
    return {"overall": dict(overall),
            "by_category": {k: dict(v) for k, v in by_category.items()}}


def harm_analysis(per_query: dict, categories: tuple) -> dict:
    """
    Rank of the first relevant chunk, dense_only vs hybrid, per query.

    RRF can demote a correctly ranked dense result. Every regression is listed
    by name; finding them is the point of the exercise.
    """
    rows, worse, better, unchanged = [], [], [], 0
    MISS = 999      # sorts below any real rank; never displayed as a number

    for qid, row in per_query.items():
        cat = row["_meta"]["category"]
        if cat not in categories:
            continue
        d_rank = row["dense_only"]["rank"]
        h_rank = row["hybrid"]["rank"]
        entry = {"id": qid, "category": cat,
                 "question": row["_meta"]["question"],
                 "dense_rank": d_rank, "hybrid_rank": h_rank}
        rows.append(entry)

        d_eff = d_rank if d_rank is not None else MISS
        h_eff = h_rank if h_rank is not None else MISS
        if h_eff > d_eff:
            worse.append(entry)
        elif h_eff < d_eff:
            better.append(entry)
        else:
            unchanged += 1

    return {"rows": rows, "worse": worse, "better": better,
            "unchanged": unchanged}


# ══════════════════════════════════════════════════════════════════════════
# Bucket 3 - unanswerable false positives
# ══════════════════════════════════════════════════════════════════════════

def unanswerable_analysis(arms: dict, queries: list, ctx: dict,
                          per_query: dict) -> dict:
    """
    How confidently does each arm answer a question with no answer?

    NOT a Hit@K measurement. The threshold is derived from the data - the
    median top-1 score this same arm produces on answerable queries - rather
    than invented, and every arm reports the score signal it actually exposes.

    This measures a refusal PRECONDITION only. It says nothing about whether
    the generator would refuse; that needs the API and was not run.
    """
    from src.vector_store import get_embeddings

    embeddings = get_embeddings()
    una = [q for q in queries if q["category"] in BUCKET_UNANSWERABLE]
    ans = [q for q in queries if q["category"] not in BUCKET_UNANSWERABLE]

    def top1_score(arm_name: str, question: str):
        """The score signal each arm actually exposes for its top-1 chunk."""
        if arm_name == "hybrid":
            docs = arms["hybrid"](question)
            return docs[0].metadata.get("rrf_score") if docs else None
        if arm_name == "bm25_only":
            hits = ctx["bm25_index"].search(question, FINAL_K)
            return hits[0][1] if hits else None
        # MMR exposes no score through the retriever interface. Cosine between
        # the query and the arm's own top-1 chunk is computed here instead and
        # is labelled as such wherever it is reported.
        docs = arms["dense_only"](question)
        if not docs:
            return None
        qv = embeddings.embed_query(question)
        dv = embeddings.embed_documents([docs[0].page_content])[0]
        return sum(a * b for a, b in zip(qv, dv))

    out: dict = {"note": "score signals differ per arm; see per-arm 'signal'",
                 "arms": {}}

    for arm_name in arms:
        ans_scores = [s for s in (top1_score(arm_name, q["question"]) for q in ans)
                      if s is not None]
        una_scores = [(q["id"], top1_score(arm_name, q["question"])) for q in una]

        ordered = sorted(ans_scores)
        threshold = ordered[len(ordered) // 2] if ordered else None

        returned_anything = sum(1 for _, s in una_scores if s is not None)
        above = sum(1 for _, s in una_scores
                    if s is not None and threshold is not None and s >= threshold)

        signal = {"hybrid": "RRF fusion score",
                  "bm25_only": "Okapi BM25 score",
                  "dense_only": "cosine(query, top-1 chunk), computed outside "
                                "the arm because MMR exposes no score"}[arm_name]

        out["arms"][arm_name] = {
            "signal": signal,
            "threshold": threshold,
            "threshold_basis": "median top-1 score on the 49 answerable queries",
            "returned_any_chunk": metrics.proportion(returned_anything, len(una)),
            "above_threshold": metrics.proportion(above, len(una)),
            "per_query": [{"id": qid, "score": s} for qid, s in una_scores],
            "answerable_median": threshold,
        }
    return out


# ══════════════════════════════════════════════════════════════════════════
# Entry points
# ══════════════════════════════════════════════════════════════════════════

def bucket3_across_builds(pinned_dir: str, queries: list,
                          extra_dirs: list, n_builds: int) -> dict:
    """
    Repeat the unanswerable analysis over several independent builds.

    Buckets 1 and 2 were measured as identical across builds, so they are
    reported from one pinned store. Bucket 3 is the only place Chroma's
    non-deterministic HNSW construction can reach a reported number, so it
    gets a range instead of a point estimate.
    """
    from src import vector_store
    from src.hybrid_retrieval import load_bm25_index

    dirs = [pinned_dir] + list(extra_dirs)
    while len(dirs) < n_builds:
        dirs.append(fresh_store(f"bucket3 build {len(dirs)+1}"))

    observations = []
    for path in dirs[:n_builds]:
        vector_store.load_vector_store.clear()
        load_bm25_index.clear()
        arms, ctx = build_arms(path)
        observations.append({
            "store": os.path.basename(path),
            "analysis": unanswerable_analysis(arms, queries, ctx, {}),
        })

    # Collapse into a range per arm per statistic.
    spread: dict = {}
    for arm in observations[0]["analysis"]["arms"]:
        above = [o["analysis"]["arms"][arm]["above_threshold"]["value"]
                 for o in observations]
        returned = [o["analysis"]["arms"][arm]["returned_any_chunk"]["value"]
                    for o in observations]
        thresholds = [o["analysis"]["arms"][arm]["threshold"]
                      for o in observations]
        per_q: dict = defaultdict(list)
        for o in observations:
            for row in o["analysis"]["arms"][arm]["per_query"]:
                per_q[row["id"]].append(row["score"])
        spread[arm] = {
            "signal": observations[0]["analysis"]["arms"][arm]["signal"],
            "above_threshold_rate": {"min": min(above), "max": max(above),
                                      "values": above},
            "returned_any_rate": {"min": min(returned), "max": max(returned),
                                   "values": returned},
            "threshold": {"min": min(thresholds), "max": max(thresholds)},
            "per_query_score_range": {
                qid: {"min": min(v), "max": max(v),
                      "spread": max(v) - min(v), "values": v}
                for qid, v in per_q.items()
            },
        }

    return {"n_builds": n_builds,
            "stores": [os.path.basename(d) for d in dirs[:n_builds]],
            "spread": spread,
            "observations": observations}


def run_full(chroma_dir: str | None, bucket3_builds: int = 1,
             extra_dirs: list | None = None) -> dict:
    queries = load_queries()
    # Resolve span anchors; validate_dataset attaches them as q["_span"].
    from benchmark.validate_dataset import attach_spans
    attach_spans(queries)

    if chroma_dir is None:
        chroma_dir = fresh_store("eval")
    arms, ctx = build_arms(chroma_dir)
    arm_names = list(arms)

    per_query = score_answerable(arms, queries, ctx)

    report = {
        "config": {
            "final_k": FINAL_K, "vector_candidate_k": VECTOR_CANDIDATE_K,
            "vector_fetch_k": VECTOR_FETCH_K, "mmr_lambda": MMR_LAMBDA,
            "bm25_candidate_k": BM25_CANDIDATE_K, "rrf_smoothing": RRF_SMOOTHING,
        },
        "store": os.path.basename(chroma_dir),
        "n_chunks": ctx["n_chunks"],
        "n_queries": len(queries),
        "bucket1_code": bucket_summary(per_query, BUCKET_CODE, arm_names),
        "bucket2_documents": bucket_summary(per_query, BUCKET_DOC, arm_names),
        "bucket3_unanswerable": unanswerable_analysis(arms, queries, ctx, per_query),
        "provenance": provenance_attribution(per_query, BUCKET_CODE),
        "harm": harm_analysis(per_query, BUCKET_CODE),
        "per_query": {k: v for k, v in per_query.items()},
    }

    if bucket3_builds > 1:
        report["bucket3_stability"] = bucket3_across_builds(
            chroma_dir, queries, extra_dirs or [], bucket3_builds
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--determinism", action="store_true",
                        help="build the store twice and compare all rankings")
    parser.add_argument("--store", default=None,
                        help="reuse an existing chroma_bench_* directory")
    parser.add_argument("--bucket3-builds", type=int, default=1,
                        help="independent builds to repeat Bucket 3 over; "
                             "Buckets 1-2 are build-invariant and use --store")
    parser.add_argument("--extra-store", action="append", default=[],
                        help="an already-built store to reuse as a Bucket 3 "
                             "observation (repeatable)")
    args = parser.parse_args()

    if args.determinism:
        report = determinism_check()
        print(json.dumps(report, indent=2, default=str)[:400] + " ...")
        ok = True
        for arm, res in report["arms"].items():
            status = "PASS" if not res["differing"] else "FAIL"
            ok = ok and not res["differing"]
            print(f"  [{status}] {arm:12} {res['identical']}/{res['total']} "
                  f"query rankings byte-identical across two builds")
            for d in res["differing"][:10]:
                print(f"        {d['id']} ({d['category']}): {d['kind']}, "
                      f"top1_same={d['top1_same']}")
        path = os.path.join(PROJECT_ROOT, "benchmark", "determinism.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nwritten: {path}")
        return 0 if ok else 1

    report = run_full(args.store, bucket3_builds=args.bucket3_builds,
                      extra_dirs=args.extra_store)
    with open(RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"written: {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
