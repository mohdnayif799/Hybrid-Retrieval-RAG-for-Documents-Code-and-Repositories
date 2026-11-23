"""
v3 retrieval evaluation: the PRODUCTION retrieval path, scored deterministically
against span-level gold. API-free: retrieval happens before generation.

    python -m benchmark.v3.evaluate [--archive PATH] [--fidelity]

What "production path" means here
---------------------------------
The arm named ``production`` is ``src.hybrid_retrieval.build_hybrid_retriever``
called exactly as ``src/rag_chain.py`` calls it, over a store built by
``src.vector_store._embed_in_batches`` (what both public store creators use).
Nothing about retrieval is re-implemented here, so a change to the shipped
retriever - constants, fusion, the README pin - is measured automatically.

Determinism
-----------
v2 found that Chroma's default HNSW settings (ef_search=100) make the dense
candidate list vary between builds and even between processes on one store.
The eval store's collection is therefore created, before production code opens
it, with HNSW parameters that make search exhaustive on this corpus
(ef_search=2000 > 1222 chunks). Measured: top-30 identical to exact cosine on
83/83 v2 queries, and the production retriever's output identical across two
independent builds on 83/83. Production keeps Chroma's defaults; ``--fidelity``
builds a second store with those defaults and reports how many production
top-5 lists differ between the two, without looking at any gold.

Arms (all in one process, one store)
------------------------------------
  production          build_hybrid_retriever(...).invoke - what the app returns
                      (final_k=5, README pin on). Recall@20 uses the same
                      retriever with final_k=20.
  production_no_pin   the same fused list without the README pin: RRF over the
                      retriever's own dense and BM25 candidates.
  dense_fallback      the retriever's MMR vector retriever alone - what
                      build_hybrid_retriever returns when BM25 is unavailable
                      (10 chunks).
  dense_plain         plain similarity search on the same store, top 20.
  bm25                the retriever's BM25 index, top 20.

Metrics, per query: Hit@1, Hit@5, MRR@5 and Recall@20 against span gold
(near-duplicates included), plus file-level Hit@5 for code queries as the v2
-comparable secondary metric. Aggregates carry paired bootstrap intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

V3_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(V3_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.v3 import dataset as D
from benchmark.v3 import gold as G
from benchmark.v3 import stats as S

RESULTS_PATH = os.path.join(V3_DIR, "results.json")
MANIFEST_PATH = os.path.join(V3_DIR, "manifest.json")
STORE_PREFIX = "chroma_bench_v3"

# Exhaustive search on this corpus: ef_search must exceed the chunk count.
EXACT_HNSW = {"space": "l2", "ef_construction": 400, "ef_search": 2000, "max_neighbors": 64}
COLLECTION = "langchain"      # langchain_community.vectorstores.Chroma's default name

ARMS = ("production", "production_no_pin", "dense_fallback", "dense_plain", "bm25")
RECALL_K = 20

# Misspelled queries whose correctly spelled twin has the same gold span; the
# per-pair rank difference is the typo effect. (msp-09, msp-10 were kept in
# review specifically for this; the identifier twins come for free.)
TYPO_TWINS = (("msp-09", "dfa-04"), ("msp-10", "dco-02"), ("msp-01", "eid-01"),
              ("msp-03", "eid-02"), ("msp-04", "eid-04"))

# Files whose content decides what is measured and how it is scored. Their
# hashes are recorded with every result; tests/test_benchmark_v3.py fails when
# the committed results no longer match them.
FINGERPRINT_FILES = (
    "src/hybrid_retrieval.py", "src/vector_store.py", "src/chunking.py",
    "src/code_languages.py", "src/repo_ingestion.py", "src/data_ingestion.py",
    "benchmark/v3/queries.yaml", "benchmark/v3/split.json",
    "benchmark/v3/gold.py", "benchmark/v3/dataset.py", "benchmark/v3/corpus.py",
    "benchmark/v3/stats.py", "benchmark/v3/evaluate.py",
)


def sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint() -> dict:
    out = {}
    for rel in FINGERPRINT_FILES:
        path = os.path.join(PROJECT_ROOT, rel)
        out[rel] = sha256_path(path) if os.path.exists(path) else None
    return out


def library_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("chromadb", "langchain-community", "langchain-core", "langchain-huggingface",
                "sentence-transformers", "transformers", "torch", "rank-bm25", "numpy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    out["python"] = sys.version.split()[0]
    return out


# ── store ────────────────────────────────────────────────────────────────────

def new_store_dir(tag: str) -> str:
    import uuid
    return os.path.join(PROJECT_ROOT, f"{STORE_PREFIX}_{tag}_{int(time.time())}_{uuid.uuid4().hex[:6]}")


def build_store(chunks: list, chroma_dir: str, exact: bool = True) -> str:
    """Embed chunks through the production path; optionally pre-create an exact collection."""
    from src import vector_store

    if exact:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        client.get_or_create_collection(COLLECTION, configuration={"hnsw": dict(EXACT_HNSW)})
    vector_store._embed_in_batches(chunks, chroma_dir)
    return chroma_dir


def store_hnsw_config(chroma_dir: str) -> dict:
    from src.vector_store import load_vector_store

    coll = load_vector_store(chroma_dir)._collection
    conf = getattr(coll, "configuration_json", None) or getattr(coll, "configuration", None) or {}
    return dict(conf.get("hnsw") or {})


# ── arms ─────────────────────────────────────────────────────────────────────

def make_arms(chroma_dir: str):
    from src.hybrid_retrieval import (
        HybridRetriever,
        build_hybrid_retriever,
        reciprocal_rank_fusion,
    )
    from src.vector_store import load_vector_store

    retriever = build_hybrid_retriever(chroma_dir)          # exactly as rag_chain.py:233
    if not isinstance(retriever, HybridRetriever):
        raise TypeError("build_hybrid_retriever fell back to vector-only; is rank_bm25 installed?")
    wide = retriever.model_copy(update={"final_k": RECALL_K})
    store = load_vector_store(chroma_dir)

    def production(q):
        return retriever.invoke(q)

    def production_wide(q):
        return wide.invoke(q)

    def production_no_pin(q):
        dense = retriever.vector_retriever.invoke(q)
        sparse = [d for d, _ in retriever.bm25_index.search(q, retriever.bm25_candidate_k)]
        return reciprocal_rank_fusion(
            {"vector": dense, "bm25": sparse},
            {"vector": retriever.vector_weight, "bm25": retriever.bm25_weight},
            smoothing=retriever.rrf_smoothing, top_k=RECALL_K)

    def dense_fallback(q):
        return retriever.vector_retriever.invoke(q)

    def dense_plain(q):
        return store.similarity_search(q, k=RECALL_K)

    def bm25(q):
        return [d for d, _ in retriever.bm25_index.search(q, RECALL_K)]

    return {
        "production": production, "production@20": production_wide,
        "production_no_pin": production_no_pin, "dense_fallback": dense_fallback,
        "dense_plain": dense_plain, "bm25": bm25,
    }, retriever


# ── scoring ──────────────────────────────────────────────────────────────────

def chunk_coverage(ranked: list, direct: dict, k: int) -> float:
    """Share of the chunks overlapping a gold span that appear in the top k."""
    if not direct:
        raise ValueError("coverage is undefined without gold chunks")
    return len(set(ranked[:k]) & set(direct)) / len(direct)


def score_query(ranked_top: list, ranked_wide: list, gold: dict, n_parts: int,
                direct: dict) -> dict:
    """
    ranked_top: what the arm returns (production: its final 5).
    ranked_wide: the arm's list for the @20 metrics (production: final_k=20).
    gold: chunk -> parts, near-duplicates included. direct: span-overlap only.
    """
    recall5 = G.recall_at(ranked_top, gold, n_parts, 5)
    return {
        "rank": G.first_rank(ranked_top, gold),
        "hit@1": float(G.hit_at(ranked_top, gold, 1)),
        "hit@5": float(G.hit_at(ranked_top, gold, 5)),
        "mrr@5": G.mrr_at(ranked_top, gold, 5),
        "recall@5": recall5,
        "all_parts@5": float(recall5 == 1.0),
        "recall@20": G.recall_at(ranked_wide, gold, n_parts, RECALL_K),
        "hit@20": float(G.hit_at(ranked_wide, gold, RECALL_K)),
        "chunk_cov@5": chunk_coverage(ranked_top, direct, 5),
        "chunk_cov@20": chunk_coverage(ranked_wide, direct, RECALL_K),
        "hit@5_direct": float(G.hit_at(ranked_top, direct, 5)),
    }


def file_hit5(docs: list, query: dict) -> float | None:
    """v2-comparable file-level Hit@5 for code queries; None elsewhere."""
    from benchmark import metrics

    rel = [e for e in (query.get("relevant") or []) if isinstance(e, str)]
    if query["domain"] != "code" or not rel:
        return None
    specs = [metrics.normalise_spec(e) for e in rel]
    return float(metrics.hit_at_k_chunks(docs[:5], specs, 5))


def run(chunks: list, parents: dict, data: dict, chroma_dir: str, splits: dict) -> dict:
    from src.hybrid_retrieval import chunk_key

    near = G.NearDuplicates(chunks)
    resolved, errors, warnings = D.validate(data, chunks, parents, near)
    if errors:
        raise RuntimeError("dataset failed validation:\n  " + "\n  ".join(errors))

    pos = {chunk_key(c): i for i, c in enumerate(chunks)}
    arms, retriever = make_arms(chroma_dir)
    vectors, embed_query = stored_vectors(chroma_dir, pos)

    def indices(docs):
        out = []
        for d in docs:
            key = chunk_key(d)
            if key not in pos:
                raise RuntimeError(f"retrieved chunk not in corpus: {key}")
            out.append(pos[key])
        return out

    per_query: dict = {}
    for q in data["queries"]:
        qid, question = q["id"], q["question"]
        lists = {name: fn(question) for name, fn in arms.items()}
        row = {"qtype": q["qtype"], "domain": q["domain"], "origin": q["origin"],
               "split": splits.get(qid, "all"),
               "pin_fired": any(d.metadata.get("retrieval") == "readme_boost"
                                for d in lists["production"]),
               "top5": {name: [f"{G.chunk_file(chunks[i])}@{chunks[i].metadata.get('start_index')}"
                               for i in indices(lists[name])[:5]] for name in ARMS},
               "top1": top1_signals(question, lists, indices, vectors, embed_query, retriever),
               "arms": {}}
        if q["qtype"] != "unanswerable":
            r = resolved[qid]
            for name in ARMS:
                top = indices(lists[name])
                wide = indices(lists["production@20"]) if name == "production" else top
                row["arms"][name] = score_query(top, wide, r.expanded, r.n_parts, r.direct)
                row["arms"][name]["file_hit@5"] = file_hit5(lists[name], q)
            row["gold_chunks"] = len(r.direct)
            row["gold_parts"] = r.n_parts
        per_query[qid] = row
    return {"per_query": per_query, "warnings": warnings,
            "retriever": {"final_k": retriever.final_k,
                          "bm25_candidate_k": retriever.bm25_candidate_k,
                          "rrf_smoothing": retriever.rrf_smoothing,
                          "vector_weight": retriever.vector_weight,
                          "bm25_weight": retriever.bm25_weight,
                          "vector_search_kwargs": dict(getattr(retriever.vector_retriever,
                                                               "search_kwargs", {}) or {}),
                          "vector_search_type": getattr(retriever.vector_retriever,
                                                        "search_type", None)}}


def stored_vectors(chroma_dir: str, pos: dict):
    """Chunk index -> the embedding production stored for it, plus the query embedder."""
    import numpy as np
    from langchain_core.documents import Document

    from src.hybrid_retrieval import chunk_key
    from src.vector_store import get_embeddings, load_vector_store

    raw = load_vector_store(chroma_dir)._collection.get(include=["embeddings", "metadatas"])
    vectors = {}
    for vec, meta in zip(raw["embeddings"], raw["metadatas"]):
        vectors[pos[chunk_key(Document(page_content="", metadata=dict(meta or {})))]] = (
            np.asarray(vec, dtype=np.float32))
    return vectors, get_embeddings().embed_query


def top1_signals(question: str, lists: dict, indices, vectors: dict, embed_query,
                 retriever) -> dict:
    """
    Confidence signals for each arm's top-1 chunk, recorded for every query so a
    refusal threshold can later be set from answerable vs unanswerable queries.

    cosine: production embedding of the query vs the stored embedding of the
    top-1 chunk - one scale for every arm. native: the score the arm itself
    exposes (RRF score for the fused arms, BM25 score for bm25; dense arms
    expose none through the retriever, so their native signal is the cosine).
    """
    import numpy as np

    qv = np.asarray(embed_query(question), dtype=np.float32)
    qv = qv / (np.linalg.norm(qv) or 1.0)
    out = {}
    for name in ARMS:
        docs = lists[name]
        if not docs:
            out[name] = {"cosine": None, "native": None}
            continue
        cv = vectors[indices(docs[:1])[0]]
        cosine = float(cv @ qv / (np.linalg.norm(cv) or 1.0))
        if name in ("production", "production_no_pin"):
            native = docs[0].metadata.get("rrf_score")
        elif name == "bm25":
            hits = retriever.bm25_index.search(question, 1)
            native = float(hits[0][1]) if hits else None
        else:
            native = cosine
        out[name] = {"cosine": cosine, "native": native}
    return out


METRICS = ("hit@1", "hit@5", "mrr@5", "recall@20", "all_parts@5", "chunk_cov@5", "chunk_cov@20")
MULTI_PART_MIN = 2
LARGE_GOLD_MIN = 4


def group_tags(row: dict) -> list[str]:
    """Reporting groups a query belongs to, besides 'all'."""
    tags = [row["qtype"]]
    if row["gold_parts"] >= MULTI_PART_MIN:
        tags.append("multi_part")
    if row["gold_chunks"] >= LARGE_GOLD_MIN:
        tags.append("gold4plus")
    return tags


def aggregate(per_query: dict, resamples: int, seed: int) -> dict:
    """Per split x domain x group: metric means with paired bootstrap CIs."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for qid, row in per_query.items():
        if row["qtype"] == "unanswerable":
            continue
        for split in (row["split"], "all") if row["split"] != "all" else ("all",):
            for dom in (row["domain"], "all"):
                for tag in [*group_tags(row), "all"]:
                    groups[(split, dom, tag)].append(qid)
    out = {}
    for (split, dom, qt), ids in sorted(groups.items()):
        ids = sorted(ids)
        idx = S.resample_matrix(len(ids), resamples, seed)
        block = {"n": len(ids), "arms": {}, "vs_production": {}}
        for arm in ARMS:
            block["arms"][arm] = {m: S.mean_ci([per_query[i]["arms"][arm][m] for i in ids], idx)
                                  for m in METRICS}
            fl = [per_query[i]["arms"][arm]["file_hit@5"] for i in ids]
            fl = [x for x in fl if x is not None]
            block["arms"][arm]["file_hit@5"] = (sum(fl) / len(fl)) if fl else None
        for arm in ARMS:
            if arm == "production":
                continue
            block["vs_production"][arm] = {
                m: S.paired_diff_ci([per_query[i]["arms"][arm][m] for i in ids],
                                    [per_query[i]["arms"]["production"][m] for i in ids], idx)
                for m in METRICS}
        out[f"{split}/{dom}/{qt}"] = block
    return out


def refusal_signal(per_query: dict) -> dict:
    """Top-1 signal distributions, answerable vs unanswerable, per split and arm."""
    import numpy as np

    out: dict = {}
    splits = sorted({r["split"] for r in per_query.values()} | {"all"})
    for split in splits:
        rows = [r for r in per_query.values() if split in ("all", r["split"])]
        block = {}
        for arm in ARMS:
            block[arm] = {}
            for label, want in (("answerable", False), ("unanswerable", True)):
                for signal in ("cosine", "native"):
                    vals = [r["top1"][arm][signal] for r in rows
                            if (r["qtype"] == "unanswerable") == want
                            and r["top1"][arm][signal] is not None]
                    block[arm][f"{label}_{signal}"] = (
                        {"n": len(vals), "median": float(np.median(vals)),
                         "p10": float(np.quantile(vals, 0.1)), "p90": float(np.quantile(vals, 0.9))}
                        if vals else None)
        out[split] = block
    return out


def typo_effect(per_query: dict) -> dict:
    """Rank of the first gold chunk for each misspelled query and its correct twin."""
    out = {}
    for typo, correct in TYPO_TWINS:
        if typo not in per_query or correct not in per_query:
            continue
        out[f"{typo}|{correct}"] = {
            arm: {"typo_rank": per_query[typo]["arms"][arm]["rank"],
                  "correct_rank": per_query[correct]["arms"][arm]["rank"]}
            for arm in ARMS}
    return out


# ── freeze guard ─────────────────────────────────────────────────────────────

def load_splits(allow_unfrozen: bool) -> tuple[dict, dict | None]:
    """{query_id: split} from the frozen manifest, or refuse."""
    if not os.path.exists(MANIFEST_PATH):
        if allow_unfrozen:
            print("WARNING: no frozen manifest; scoring an UNFROZEN query set, all queries in 'all'.")
            return {}, None
        raise SystemExit(
            "refusing to score: benchmark/v3/manifest.json does not exist, so the query set is "
            "not frozen. Freeze it first, or pass --allow-unfrozen for a development run.")
    from benchmark.v3 import freeze
    manifest = freeze.verify()          # raises if any frozen file changed
    return freeze.load_split_assignments(), manifest


def fidelity(chunks: list, data: dict, exact_dir: str) -> dict:
    """Production top-5 on the exact store vs a store with Chroma's default HNSW."""
    from src.hybrid_retrieval import build_hybrid_retriever, chunk_key

    default_dir = build_store(chunks, new_store_dir("default"), exact=False)
    a = build_hybrid_retriever(exact_dir)
    b = build_hybrid_retriever(default_dir)
    same, differ = 0, []
    for q in data["queries"]:
        ka = [chunk_key(d) for d in a.invoke(q["question"])]
        kb = [chunk_key(d) for d in b.invoke(q["question"])]
        if ka == kb:
            same += 1
        else:
            differ.append(q["id"])
    return {"identical": same, "total": len(data["queries"]), "differ": differ,
            "default_store_hnsw": store_hnsw_config(default_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", help="local psf/requests tarball at the pinned commit")
    parser.add_argument("--store", help="reuse an existing exact eval store directory")
    parser.add_argument("--out", default=RESULTS_PATH)
    parser.add_argument("--resamples", type=int, default=S.DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=S.DEFAULT_SEED)
    parser.add_argument("--fidelity", action="store_true",
                        help="also compare against a default-HNSW store (no gold involved)")
    parser.add_argument("--allow-unfrozen", action="store_true",
                        help="development only: score without a frozen manifest")
    args = parser.parse_args()

    splits, manifest = load_splits(args.allow_unfrozen)

    from benchmark.v3.corpus import load_corpus

    started = time.time()
    chunks, parents, corpus_meta = load_corpus(args.archive)
    data = D.load()
    chroma_dir = args.store or build_store(chunks, new_store_dir("exact"), exact=True)
    result = run(chunks, parents, data, chroma_dir, splits)
    report = {
        "config": result["retriever"],
        "store": {"dir": os.path.basename(chroma_dir), "hnsw": store_hnsw_config(chroma_dir)},
        "corpus": corpus_meta,
        "fingerprint": fingerprint(),
        "manifest_sha256": manifest.get("manifest_sha256") if manifest else None,
        "frozen": manifest is not None,
        "libraries": library_versions(),
        "bootstrap": {"resamples": args.resamples, "seed": args.seed, "alpha": S.ALPHA},
        "aggregates": aggregate(result["per_query"], args.resamples, args.seed),
        "unanswerable": {
            "n": sum(1 for r in result["per_query"].values() if r["qtype"] == "unanswerable"),
            "pin_fired": sum(1 for r in result["per_query"].values()
                             if r["qtype"] == "unanswerable" and r["pin_fired"]),
        },
        "refusal_signal": refusal_signal(result["per_query"]),
        "typo_effect": typo_effect(result["per_query"]),
        "pin_fired_answerable": sum(1 for r in result["per_query"].values()
                                    if r["qtype"] != "unanswerable" and r["pin_fired"]),
        "dataset_warnings": result["warnings"],
        "per_query": result["per_query"],
        "seconds": round(time.time() - started, 1),
    }
    if args.fidelity:
        report["fidelity_default_hnsw"] = fidelity(chunks, data, chroma_dir)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, sort_keys=False, default=str)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
