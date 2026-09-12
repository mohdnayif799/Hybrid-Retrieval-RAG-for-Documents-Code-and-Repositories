# Retrieval Evaluation — Results

Retrieval-only evaluation of the hybrid (MMR + BM25 / RRF) retriever against
dense-only and BM25-only baselines. **No Gemini API call was made anywhere in
this workflow**; retrieval runs entirely before generation, so none is needed.

Every number below was produced by code that actually executed. Where something
could not be measured, it says so.

---

## 1. Methodology

### 1.1 Corpus

| Part | Identity | Chunks |
|---|---|---|
| Repository | `psf/requests` @ `dae7ef63b4df6eded86637f251fc4e3a06c3b479` | 1111 |
| PDF | `Deep_Learning_Unit1_Quick_Notes.pdf` · `sha256 974bc4248d10cbd6…` · 2 pages | 4 |
| Text | `space_planets_moons_advanced.txt` · `sha256 581fd18b9738b3ef…` · 6,896 B | 8 |
| | **Total** | **1123** |

Pinned to a **commit SHA, never a branch** — a branch makes the benchmark
irreproducible. Archive `sha256 de310df9b97e0c2b…`, 3,333,774 bytes, cached at
`benchmark/.cache/` and hash-checked on every load, so a substituted or
corrupted corpus fails loudly rather than silently changing the numbers.

Ingestion used the **real pipeline with no mocking**: `parse_repo_url` →
`resolve_repository` → `download_archive` → `read_archive` →
`repo_files_to_documents` → `chunk_documents`, and `load_documents` →
`chunk_documents` for the documents. Embeddings are real `all-MiniLM-L6-v2`
(384-dim, CPU), persisted to a real Chroma store.

Of 128 archive members, 95 files were kept; 33 skipped (26 unsupported
extension, 6 excluded filename, 1 empty). Repo chunking is language-aware at
800/100; document chunking is 1000/200. **Chunk distribution is heavily
skewed** — `tests/test_requests.py` alone yields 180 chunks (16% of the corpus)
and `HISTORY.md` 108, while 41 files yield exactly one each.

The benchmark store lives under a third namespace, `chroma_bench_*`, matched by
neither glob in `store_paths.py`: `chroma_db_*` is app.py's per-chat namespace,
and `chroma_eval_*` is deleted wholesale by `stale_eval_store_dirs()`.

### 1.2 Ground truth

56 queries in `benchmark/queries.yaml`. Ground truth is at **file/page level,
never chunk level**, so it survives a change to chunk size.

| Category | n | | Category | n |
|---|---|---|---|---|
| exact_identifier | 8 | | cross_file | 7 |
| paraphrase | 8 | | doc_factual | 5 |
| conceptual | 8 | | doc_conceptual | 5 |
| literal_string | 8 | | unanswerable | 7 |

`doc_factual` and `doc_conceptual` are 5 rather than 7 — a decision made
**before any measurement**. The document half contains only **three distinct
locators in the entire corpus** (PDF page 0, PDF page 1, the whole .txt), so
7+7 would force most document questions to share identical ground truth. That
weight went to the code categories instead.

**Authoring protocol.** Every question was written from an information need
first and the answer located afterwards. Repo questions were drawn from
knowledge of the `requests` public API and its documentation, never by reading
a chunk and paraphrasing it — that would make the question inherit the chunk's
vocabulary, which is precisely what BM25 scores. `exact_identifier` and
`literal_string` deliberately use the literal token, because that is how people
actually search. All 56 locators were verified to resolve to chunks that exist;
18 span anchors were resolved to character ranges derived from the real files
rather than hand-typed.

### 1.3 Arms

All three share **one** MMR retriever object, so `fetch_k` and `lambda_mult`
cannot drift between arms:

```python
vector_retriever = store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.6})

dense_only  = lambda q: vector_retriever.invoke(q)[:5]      # truncation
bm25_only   = lambda q: bm25_index.search(q, 5)
hybrid      = HybridRetriever(vector_retriever=vector_retriever, ...)  # same object
```

`dense_only` is a **truncation of the same fetch_k=30 selection**, not a
separate `k=5` call. MMR is greedy, so the first 5 of a k=10 selection equal a
k=5 selection only when `fetch_k` is unchanged.

Shipped configuration: `FINAL_K=5`, `VECTOR_CANDIDATE_K=10`, `VECTOR_FETCH_K=30`,
`MMR_LAMBDA=0.6`, `BM25_CANDIDATE_K=10`, `RRF_SMOOTHING=60`,
`VECTOR_WEIGHT=BM25_WEIGHT=0.5`.

### 1.4 Why three buckets rather than one aggregate

The corpus is 1111 code chunks against 12 document chunks on two unrelated
topics. A document query therefore competes with a pool ~90× smaller than a
code query does, and nothing in `requests` concerns neural networks or planets.
Folding those two regimes into a single Hit@5 would average a ceiling effect
together with a real measurement and report a misleading number even though
every individual measurement is correct.

---

## 2. Verification performed before measuring

### 2.1 Determinism — **FAILED as a strict byte-identity test**

Two independent builds, 56 queries × 3 arms:

| Arm | Identical rankings | Differing |
|---|---|---|
| `bm25_only` | **56 / 56** | — |
| `hybrid` | 52 / 56 | `par-04`, `dco-05`, `una-04`, `una-06` |
| `dense_only` | 51 / 56 | `par-04`, `con-08`, `dco-05`, `una-04`, `una-06` |

| Query | Arm(s) | Kind | first-relevant rank A → B |
|---|---|---|---|
| `par-04` | dense, hybrid | different chunk at rank 5 | 1 → 1 |
| `con-08` | dense | different chunk at rank 5 | 1 → 1 |
| `dco-05` | dense | different chunk at rank 5 | 1 → 1 |
| `dco-05` | hybrid | **reordered ties** (ranks 3↔4, same chunks) | 1 → 1 |
| `una-04` | dense, hybrid | different chunks ranks 3–5 | n/a — top-1 same |
| `una-06` | dense, hybrid | different chunks from rank 1 | n/a — **top-1 differs** |

**Root cause, diagnosed rather than assumed.** A single store queried three
times returns byte-identical results, so there is no query-time randomness.
`export_chunks()` returns the identical set *and order* across builds, which is
why `bm25_only` is perfectly stable. The raw ANN `similarity_search(k=30)`
pool — *upstream of MMR* — differs across builds, overlapping 24–29 of 30 on
the affected queries. **Chroma's HNSW graph construction is non-deterministic
across builds**; the candidate pool shifts and MMR selects from it. No project
code is at fault.

**Measured impact.** The full evaluation was run on both builds and every
reported value diffed:

```
BUCKET 1 (code)        : 0 differing values
BUCKET 2 (documents)   : 0 differing values
PROVENANCE             : 0 differing
HARM                   : 0 differing
BUCKET 3 (unanswerable): 1 differing value
    dense_only[una-06] cosine   A=0.2612  B=0.3133
```

Buckets 1 and 2 are therefore reported from one pinned store
(`chroma_bench_1788912379_36bfad5f`). Bucket 3, the only bucket the instability
can reach, was repeated over **5 independent builds**; results in §5.

### 2.2 Other checks

- **Dense-only construction** — PASS. Identical by construction (shared object).
- **Unanswerable scoring** — PASS. `hit_at_k`, `mrr_at_k`, `recall_at_k` all
  **raise `ValueError`** on an empty relevant set rather than returning 0, and
  the scoring loop skips `unanswerable` entirely. A fabricated 0 would drag
  every aggregate down by the number of unanswerable queries.
- **Cross-file spot check** — PASS. All 7 verified against real source; none is
  a single-file answer mislabelled. Two required locators are single-chunk
  files (`certs.py`, `__version__.py`), which makes Recall@5 structurally hard
  there — a property, not a defect.

### 2.3 Bias audit

Token overlap between each question and its ground-truth text, using the
project's own `hybrid_retrieval.tokenize` — the function that actually decides
BM25 candidacy.

| Category | n | min | median | max | mean |
|---|---|---|---|---|---|
| exact_identifier | 8 | 1.00 | 1.00 | 1.00 | **1.00** |
| literal_string | 8 | 0.67 | 1.00 | 1.00 | **0.96** |
| doc_factual | 5 | 0.50 | 0.60 | 0.80 | 0.61 |
| doc_conceptual | 5 | 0.25 | 0.44 | 0.60 | 0.45 |
| cross_file | 7 | 0.18 | 0.40 | 0.44 | 0.35 |
| paraphrase | 8 | 0.10 | 0.29 | 0.56 | **0.30** |
| conceptual | 8 | 0.10 | 0.29 | 0.50 | **0.29** |
| unanswerable | 7 | 0.17 | 0.43 | 0.60 | 0.37 |

`exact_identifier` at 1.00 and `literal_string` at 0.96 are correct by
construction. The load-bearing result is **`paraphrase` 0.30 and `conceptual`
0.29** — those questions do not carry their answers' vocabulary, so a hybrid
win there would be real rather than an artefact.

The audit caught one genuine error: an `unanswerable` query about `async`/
`await` scored 1.00 best-in-corpus overlap, and inspection showed
`docs/community/recommended.rst` **does** discuss it via Twisted. The original
verification grep had searched `"await "` with a trailing space and missed the
RST markup. It was replaced with a server-sent-events question (verified 0
hits). One `doc_conceptual` query is flagged at 0.60 and was reviewed and kept:
the overlap is carried by the entity's own name ("black hole", "star"), which
no natural phrasing avoids.

---

## 3. Bucket 1 — Code corpus (n = 39)

The real dense-vs-hybrid test, against the 1111-chunk repository.

### 3.1 Per category (Hit@5, 95% Wilson CI) — reported before the aggregate

| Category | n | dense_only | bm25_only | hybrid |
|---|---|---|---|---|
| exact_identifier | 8 | 87.5% [52.9, 97.8] | 87.5% [52.9, 97.8] | 87.5% [52.9, 97.8] |
| paraphrase | 8 | 87.5% [52.9, 97.8] | 75.0% [40.9, 92.9] | 87.5% [52.9, 97.8] |
| conceptual | 8 | 87.5% [52.9, 97.8] | 62.5% [30.6, 86.3] | **100.0% [67.6, 100.0]** |
| literal_string | 8 | 100.0% [67.6, 100.0] | 100.0% [67.6, 100.0] | 100.0% [67.6, 100.0] |
| cross_file | 7 | **85.7% [48.7, 97.4]** | 42.9% [15.8, 75.0] | 57.1% [25.0, 84.2] |

### 3.2 Aggregate

| Metric | dense_only | bm25_only | hybrid |
|---|---|---|---|
| Hit@1 | **53.8% [38.6, 68.4]** | 38.5% [24.9, 54.1] | 48.7% [33.9, 63.8] |
| Hit@3 | 79.5% [64.5, 89.2] | 66.7% [51.0, 79.4] | **84.6% [70.3, 92.8]** |
| Hit@5 | **89.7% [76.4, 95.9]** | 74.4% [58.9, 85.4] | 87.2% [73.3, 94.4] |
| MRR@5 | **0.678** | 0.529 | 0.642 |
| span Hit@5 (n=13) | 69.2% [42.4, 87.3] | 46.2% [23.2, 70.9] | **76.9% [49.7, 91.8]** |
| cross_file Recall@5 | **0.452** | 0.214 | 0.381 |

### 3.3 McNemar, dense vs hybrid

```
dense won 2   hybrid won 1   discordant 3   p = 1.0000
```

**Only 3 discordant pairs — the comparison is underpowered.** With fewer than
6, no arrangement of outcomes can reach p < 0.05. This p-value should be read
as "this experiment cannot distinguish these arms," not as evidence of
equivalence.

**Supplementary: exact two-sided sign test on rank movement.** McNemar counts
only Hit@5 flips (3 observations). The rank-change tally in §3.5 is
finer-grained and supports a test with 21:

| | |
|---|---|
| Hybrid worse / better / tied | 13 / 8 / 18 |
| Non-tied observations | 21 |
| **p (exact, two-sided)** | **0.3833** |
| `scipy.stats.binomtest` | 0.3833 — agrees to 1e-12 |
| Significant at α = 0.05 | **No** |

This **supplements** the McNemar figure above; it does not replace it. **It
does not change the conclusion** — seven times as many observations still
cannot distinguish the arms.

### 3.4 Provenance — which retriever surfaced hybrid's first relevant chunk

Of hybrid's 34 successes in Bucket 1:

| Origin | Count | Share |
|---|---|---|
| `vector+bm25` (both) | 23 | 68% |
| `vector` only | 7 | 21% |
| `bm25` only | **4** | **12%** |

| Category | vector+bm25 | vector | bm25 |
|---|---|---|---|
| exact_identifier | 5 | 1 | 1 |
| paraphrase | 5 | 1 | 1 |
| conceptual | 4 | 3 | 1 |
| literal_string | **8** | 0 | 0 |
| cross_file | 1 | 2 | 1 |

BM25 was the **sole** discoverer in 4 of 34 successes. `literal_string` is 8/8
`vector+bm25` — on this corpus the dense retriever finds the constants too, so
BM25's contribution there is corroboration rather than rescue.

### 3.5 Harm analysis — every regression named

**8 improved · 13 regressed · 18 unchanged.**

Regressions (rank of first relevant chunk, dense → hybrid):

| Query | Category | dense | hybrid | Cost |
|---|---|---|---|---|
| `crs-03` | cross_file | 4 | **miss** | **lost the answer** |
| `crs-06` | cross_file | 4 | **miss** | **lost the answer** |
| `con-04` | conceptual | 3 | 5 | rank only |
| `crs-04` | cross_file | 1 | 3 | rank only |
| `eid-02` | exact_identifier | 2 | 3 | rank only |
| `eid-05` | exact_identifier | 2 | 3 | rank only |
| `par-04` | paraphrase | 2 | 3 | rank only |
| `con-05` | conceptual | 2 | 3 | rank only |
| `eid-01` | exact_identifier | 1 | 2 | rank only |
| `eid-03` | exact_identifier | 1 | 2 | rank only |
| `eid-08` | exact_identifier | 1 | 2 | rank only |
| `con-06` | conceptual | 1 | 2 | rank only |
| `con-08` | conceptual | 1 | 2 | rank only |

Improvements:

| Query | Category | dense | hybrid |
|---|---|---|---|
| `con-01` | conceptual | **miss** | 3 |
| `par-08` | paraphrase | 3 | 1 |
| `con-02` | conceptual | 4 | 2 |
| `crs-01` | cross_file | 5 | 3 |
| `par-03` | paraphrase | 3 | 2 |
| `con-03` | conceptual | 2 | 1 |
| `lit-01` | literal_string | 2 | 1 |
| `lit-08` | literal_string | 2 | 1 |

### 3.6 Mechanism behind the regressions

This is the most actionable finding in the evaluation, and it is arithmetic,
not noise. With `VECTOR_WEIGHT = BM25_WEIGHT = 0.5` and `RRF_SMOOTHING = 60`, a
dense chunk at rank *r* scores `0.5/(60+r)` and a BM25 chunk at rank *r* scores
exactly the same. Ties break toward the dense list, so when the two lists do
not overlap the fused order is **strict interleaving**:

```
d1, b1, d2, b2, d3   ← the entire top-5
```

**Dense positions 4 and 5 are always evicted** whenever BM25 contributes ≥2
non-overlapping candidates. Both lost queries had their only relevant chunk at
dense rank 4, and the observed hybrid output matches the predicted pattern
exactly:

- `crs-03` — dense: `utils.py` at rank 4. Hybrid slots: `vector, bm25, vector,
  bm25, vector`. `utils.py` evicted; `docs/user/advanced.rst` (BM25) took slot 4.
- `crs-06` — dense: `sessions.py` at ranks 4 **and** 5. Hybrid inserted
  `.github/CONTRIBUTING.md` and `.github/SECURITY.md` (BM25 lexical noise) at
  slots 2 and 4, displacing both.

`cross_file` is hit hardest because its answers are spread across several files
and therefore sit deeper in the dense ranking, exactly in the evicted zone.

---

## 4. Bucket 2 — Document retrieval (n = 10)

> **Small corpus, ceiling effect expected — not used for the dense-vs-hybrid
> comparison.** 12 document chunks against 1111 code chunks, on topics with no
> lexical or semantic overlap with the repository. All arms scoring identically
> here is the **expected and correct** result, not a finding.

| Category | n | dense_only | bm25_only | hybrid |
|---|---|---|---|---|
| doc_factual | 5 | 100.0% [56.6, 100.0] | 100.0% [56.6, 100.0] | 100.0% [56.6, 100.0] |
| doc_conceptual | 5 | 100.0% [56.6, 100.0] | 100.0% [56.6, 100.0] | 100.0% [56.6, 100.0] |

| Metric | dense_only | bm25_only | hybrid |
|---|---|---|---|
| Hit@1 | 90.0% [59.6, 98.2] | 100.0% [72.2, 100.0] | 100.0% [72.2, 100.0] |
| Hit@5 | 100.0% [72.2, 100.0] | 100.0% [72.2, 100.0] | 100.0% [72.2, 100.0] |
| MRR@5 | 0.933 | 1.000 | 1.000 |

McNemar: 0 discordant pairs, p = 1.0000. No difference is claimed or
detectable. The prediction that this bucket would ceiling out was made in
writing before it was run, and it did.

---

## 5. Bucket 3 — Unanswerable false-positive rate (n = 7)

Not a Hit@K measurement. Each arm reports the score signal it actually exposes,
against a threshold **derived from the data** — that arm's own median top-1
score over the 49 answerable queries — rather than an invented constant.

| Arm | Signal | Threshold | Returned any chunk | ≥ threshold (FP rate) |
|---|---|---|---|---|
| `dense_only` | cosine(query, top-1)¹ | 0.5177 | 7/7 = 100% [65, 100] | **0/7 = 0% [0, 35]** |
| `bm25_only` | Okapi BM25 score | 14.5379 | 7/7 = 100% [65, 100] | 1/7 = 14% [3, 51] |
| `hybrid` | RRF fusion score | 0.0161 | 7/7 = 100% [65, 100] | **2/7 = 29% [8, 64]** |

¹ MMR exposes no score through the retriever interface. This cosine is computed
*outside* the arm against the arm's own top-1 chunk and is labelled as such
wherever it appears — it is not the quantity MMR ranked by.

**All three arms return 5 chunks for every unanswerable question.** Nothing in
the retrieval layer abstains. The only usable refusal signal is score
magnitude, and hybrid's is the least discriminating of the three: 2 of 7
unanswerable queries produce an RRF score at or above the median score of a
genuinely answerable one.

### Stability across 5 independent builds

| | |
|---|---|
| Builds | 5 |
| Values that varied | **1 of 21 per-query scores** |
| The one that varied | `dense_only[una-06]` cosine, 0.2612 – 0.3133 (spread 0.052) |
| Thresholds | identical across all 5 builds |
| **All reported rates** | **identical across all 5 builds** |

Chroma's build non-determinism moves one raw score and no reported proportion.

---

## 6. Citation verification (Phase 4)

Zero relevance judgments required — the source file is the ground truth.

| Check | Result |
|---|---|
| Repo `start_line` accuracy | **1111 / 1111 verified, 0 mismatched** |
| PDF chunk text present on cited page | **4 / 4 verified, 0 mismatched** |
| Text chunk at recorded offset | **8 / 8 verified, 0 mismatched** |
| Citation group collisions | **0** across 97 distinct groups |
| PPTX slide verification | **not measured** — no PPTX in the corpus |

`start_line` was checked two ways: recomputed from `start_index` against the
real file text, *and* by confirming the chunk's first line appears on that line
of the file.

The grouping check is genuinely exercised rather than vacuous — the corpus
contains 6 basenames shared by multiple paths (`Makefile` ×8, `README.md` ×4,
`LICENSE` ×3, `__init__.py` ×2, `compat.py` ×2, `utils.py` ×2). Under
basename grouping all 8 Makefiles would collapse into one citation; under
`citation_group_key` they stay distinct.

This is the most methodologically solid section here: 1123 objective checks,
no judgment calls, zero failures.

---

## 7. Tests

| | |
|---|---|
| Baseline before this work | **153 passed**, 58 subtests |
| New tests added (`tests/test_benchmark_metrics.py`) | **37** |
| After | **190 passed**, 58 subtests |

Tests cover locator identity, rank metrics, the undefined-on-empty-ground-truth
guards, Wilson intervals and McNemar's exact test — on **synthetic fixtures
only**, never on benchmark results, which move whenever the corpus or pin
moves. `mcnemar_exact` is cross-checked against `scipy.stats.binomtest` over an
81-cell grid; agreeing with an independent implementation is worth more than
agreeing with my own arithmetic.

Writing these tests found a real defect in the harness: `wilson_ci(0, n)`
returned a lower bound of 2.8e-17 instead of exactly 0, so the interval did not
contain its own point estimate. Fixed by pinning the exact endpoints at p=0 and
p=1. Re-running the evaluation afterwards changed no reported figure.

---

## 8. Limitations

- **Sample size.** Bucket 1 is 39 queries; Bucket 2 is 10; Bucket 3 is 7. The
  confidence intervals are correspondingly wide — a per-category Hit@5 of 87.5%
  carries a CI of [52.9, 97.8]. Most differences visible in the tables are well
  inside each other's intervals.
- **Underpowered paired test.** 3 discordant pairs. p = 1.0 here means the
  experiment cannot distinguish the arms, not that they are equivalent.
- **Human-authored questions carry annotation bias.** The protocol and the bias
  audit reduce it; they do not eliminate it. I wrote both the questions and the
  code being evaluated, which is the blind spot the audit exists to expose and
  cannot fully close.
- **File-level ground truth does not measure passage usefulness.** A query
  scores a hit when a chunk from the right file is retrieved. It says nothing
  about whether that chunk was the most useful passage in the file. The
  span-level metric addresses this for the 13 queries carrying a span anchor,
  and no further.
- **Corpus imbalance.** 1111 code chunks vs 12 document chunks. Bucket 2 is a
  sanity check, not a comparison.
- **Chunk-count skew.** `tests/test_requests.py` supplies 16% of all chunks;
  41 files supply one each. File-level Hit@5 is structurally easier for large
  files.
- **One repository, one embedding model, one configuration.** `psf/requests` is
  a well-documented Python library with unusually clear naming. Results may not
  transfer to a JavaScript monorepo, sparse documentation, or another embedder.
- **Build non-determinism.** Documented in §2.1 and bounded by measurement, but
  present.
- **PPTX citation verification could not be run** — no PPTX in the corpus.
- **This measures RETRIEVAL ONLY.** No answer was generated. No API was called.

---

## 9. What cannot be concluded

**Hybrid retrieval was not shown to beat dense-only retrieval on this
benchmark.** On Bucket 1 aggregate Hit@5, hybrid scored 87.2% against
dense-only's 89.7% — hybrid is *lower*, and the difference is not statistically
significant (McNemar p = 1.0, 3 discordant pairs). Hit@1 and MRR@5 also favour
dense-only. Hybrid leads on Hit@3 (84.6% vs 79.5%) and span Hit@5 (76.9% vs
69.2%), and those differences are equally non-significant.

**The correct summary is that this experiment cannot distinguish the two arms
on aggregate.** It is not evidence that they are equivalent, and it is not
evidence that hybrid helps.

Specifically, nothing here establishes:

- that hybrid retrieval improves **answer** quality — no answers were generated;
- that hybrid helps or hurts on any corpus other than this one;
- that the `conceptual` result (hybrid 100% vs dense 87.5%) is real — it rests
  on a single query, `con-01`, which dense missed entirely;
- that the `cross_file` deficit (57.1% vs 85.7%) generalises — it rests on 7
  queries and two evictions, though §3.6 gives it a mechanism, which is
  stronger than the count alone;
- that the Bucket 3 false-positive rates differ meaningfully between arms — the
  intervals [0, 35], [3, 51] and [8, 64] overlap heavily;
- that any arm would cause a downstream model to hallucinate or refuse. Bucket
  3 measures a refusal **precondition** only. Whether the generator actually
  refuses requires the API and was not run.

### What the evidence does support

Two findings rest on mechanism rather than on a p-value, and are the more
useful outputs of this work:

1. **BM25 was the sole discoverer of the answer in only 4 of hybrid's 34
   successes (12%)** on this corpus, and in 0 of 8 `literal_string` queries —
   the category it was expected to own. On a well-named Python library, the
   dense retriever already finds the identifiers and constants.

2. **Equal-weight RRF at `FINAL_K=5` structurally evicts dense ranks 4 and 5**
   whenever BM25 supplies two non-overlapping candidates (§3.6). This is
   arithmetic, it predicted both observed losses in advance, and it points at
   concrete follow-ups: raise `FINAL_K`, lower `BM25_WEIGHT`, or require a
   minimum BM25 score before a lexical candidate may enter fusion. **None of
   those changes was made or evaluated here** — this is a measurement, not a
   tuning exercise.

   **Follow-up.** Those configurations were subsequently tested as
   *hypotheticals*, with no shipped file changed — see
   **[TUNING.md](TUNING.md)**. Raising `final_k` to 8 recovers both named
   losses with zero new harm in Bucket 1 and no change to the Bucket 3
   false-positive rate, but costs 60% more context, is not statistically
   significant, and rests on the same two queries that motivated it. A
   per-query median BM25 gate recovered neither loss and slightly reduced
   Hit@1, Hit@3 and MRR@5. **The data does not support changing the default.**
   TUNING.md §B.8 also records a reproducibility finding that extends §2.1:
   opening a Chroma store rewrites `chroma.sqlite3`, and retrieval can differ
   across processes on the same store directory.

---

## 10. Reproducing

```bash
python -m benchmark.validate_dataset      # locators + bias audit
python evaluate_retrieval.py --determinism
python evaluate_retrieval.py --store <chroma_bench_dir> --bucket3-builds 5
python evaluate_citations.py
python -m pytest tests/ -q
```

Artifacts: `benchmark/results.json`, `benchmark/determinism.json`,
`benchmark/citations.json`, `benchmark/corpus_manifest.json`.
