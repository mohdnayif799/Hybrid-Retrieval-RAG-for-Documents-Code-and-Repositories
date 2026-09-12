# Configuration Sweep — Testing the Diagnosed RRF Mechanism

Supplement to [RESULTS.md](RESULTS.md). Extends §3.5 (harm analysis) and §3.6
(the interleaving mechanism).

> **No shipped file was changed.** `src/hybrid_retrieval.py`, `app.py` and every
> shipped constant are untouched. Every configuration below was assembled from
> building blocks the project already exports. **This is a measurement of
> hypothetical configurations. Adopting any of them is a separate decision for
> the project owner, and is not a conclusion of this work.**
>
> No Gemini API call was made.

---

## Part A — Sign test on the existing harm tally

RESULTS.md §3.3 reported McNemar on Hit@5 flips: 3 discordant pairs, p = 1.0.
The rank-change tally in §3.5 is finer-grained — it uses every query whose rank
moved, not only those that crossed the top-5 boundary — so it supports a test
with 21 observations instead of 3.

Exact two-sided sign test on the non-tied rank changes:

| | |
|---|---|
| Hybrid worse than dense | 13 |
| Hybrid better than dense | 8 |
| Tied (excluded) | 18 |
| Non-tied observations | **21** |
| **p (exact, two-sided)** | **0.3833** |
| `scipy.stats.binomtest` | 0.3833 — **agrees to 1e-12** |
| Significant at α = 0.05 | **No** |

**This does not change the conclusion.** p = 0.3833 is nowhere near
significance. The finer-grained test uses seven times as many observations as
McNemar and still cannot distinguish the arms. Both figures now stand in the
report: McNemar for the Hit@5 outcome the generator actually experiences, the
sign test for the underlying rank movement. Neither supports a claim that
hybrid helps or hurts on aggregate.

---

## Part B — Pre-registered configuration sweep

### B.1 Configurations, fixed before anything was run

| Config | `final_k` | BM25 gate | Rationale, stated in advance |
|---|---|---|---|
| `C0_baseline` | 5 | none | The shipped configuration. Reused from `results.json`. |
| `C1_k6` | 6 | none | Minimal response to the "relevant chunk sat at dense rank 4" finding. |
| `C2_k8` | 8 | none | Tests whether the wider 13-query "worse" group recovers, not just the 2 total losses. |
| `C3_gated` | 5 | drop BM25 candidates below the median of **that query's own** candidate scores | Self-normalising; needs no global constant. |

**On C3's rule.** BM25 scores are corpus- and query-relative with no absolute
meaning, so any fixed cutoff would be a magic number fitted to one corpus.
Taking the median of the query's *own* candidate list requires no constant and
**cannot be reverse-engineered from the two known failures** — the rule was
specified without reference to which documents it happens to exclude. Ties at
the median are kept, so the gate can never empty a non-empty candidate list.

### B.2 A prediction recorded before running

`reciprocal_rank_fusion` scores every candidate, sorts, and only then applies
`[:top_k]`. `final_k` therefore **truncates but never reorders**. Two
consequences were written down in advance:

1. Hit@1, Hit@3, Hit@5 and MRR@5 must be **identical** for C0, C1 and C2. The
   decision-relevant metric for those configs is **Hit@final_k** — whether a
   relevant chunk is among the chunks the generator actually receives.
2. From the interleaving `d1, b1, d2, b2, d3, b3, d4, …`, a chunk at dense
   rank 4 lands at fused **position 7**. So `C1_k6` should **not** recover the
   two losses and `C2_k8` should.

Both predictions held exactly. They are recorded here because a prediction that
survives contact with the data is stronger evidence for the mechanism than the
same numbers presented after the fact.

### B.3 Bucket 1 results (n = 39, 95% Wilson CIs)

| Config | Hit@1 | Hit@3 | Hit@5 | Hit@`final_k` | MRR@5 |
|---|---|---|---|---|---|
| `C0_baseline` | 48.7% [33.9, 63.8] | 84.6% [70.3, 92.8] | 87.2% [73.3, 94.4] | 87.2% (k=5) | 0.642 |
| `C1_k6` | 48.7% [33.9, 63.8] | 84.6% [70.3, 92.8] | 87.2% [73.3, 94.4] | 87.2% (k=6) | 0.642 |
| `C2_k8` | 48.7% [33.9, 63.8] | 84.6% [70.3, 92.8] | 87.2% [73.3, 94.4] | **92.3% [79.7, 97.3]** (k=8) | 0.642 |
| `C3_gated` | **46.2%** [31.6, 61.4] | **82.1%** [67.3, 91.0] | 87.2% [73.3, 94.4] | 87.2% (k=5) | **0.627** |

C0/C1/C2 are identical on the @1/@3/@5 metrics, exactly as predicted — raising
`final_k` cannot move them. C3 is the only config that changes the ranking, and
it moves Hit@1, Hit@3 and MRR@5 **downward**.

### B.4 Recovery of the two named total losses

| Config | `crs-03` | `crs-06` |
|---|---|---|
| `C0_baseline` | ❌ lost | ❌ lost |
| `C1_k6` | ❌ lost | ❌ lost |
| `C2_k8` | ✅ **recovered at rank 7** | ✅ **recovered at rank 7** |
| `C3_gated` | ❌ lost | ❌ lost |

Both recover at rank 7 under C2, precisely as the interleaving arithmetic
predicted. `C1_k6` fails because position 6 is `b3`, not `d4` — a six-slot
window is one short.

### B.5 New harm introduced, measured against C0

| Config | Better | Worse | Unchanged | Detail |
|---|---|---|---|---|
| `C1_k6` | 0 | **0** | 39 | no query changed at all |
| `C2_k8` | 2 | **0** | 37 | `crs-03` miss→7, `crs-06` miss→7 |
| `C3_gated` | 1 | **1** | 37 | `eid-02` 3→2 · **`par-08` 1→4** |

`C2_k8` introduces **no new harm** in Bucket 1. `C3_gated` is net-neutral at
best: it gains one rank on `eid-02` and loses three on `par-08`, a paraphrase
query that C0 answered at rank 1.

### B.6 Bucket 3 tradeoff check (n = 7)

| Config | Threshold | Returned any chunk | Above threshold (FP rate) |
|---|---|---|---|
| `C0_baseline` | 0.016133 | 7/7 = 100% | 2/7 = 28.6% [8.2, 64.1] |
| `C1_k6` | 0.016133 | 7/7 = 100% | 2/7 = 28.6% [8.2, 64.1] |
| `C2_k8` | 0.016133 | 7/7 = 100% | 2/7 = 28.6% [8.2, 64.1] |
| `C3_gated` | 0.016133 | 7/7 = 100% | 2/7 = 28.6% [8.2, 64.1] |

**No configuration changes the false-positive rate.** For C0/C1/C2 this is
expected — `final_k` cannot alter the top-1 RRF score. For C3 it is a genuine
measurement: the gate *does* move one score (`una-01`, 0.015659 → 0.008197) but
that query was already below threshold, so the rate is unmoved. No config trades
Bucket 1 recall for extra low-confidence material.

**Gate activity** (confirming C3 is not a no-op): across all 56 queries the gate
saw 548 BM25 candidates and dropped **274 (50.0%)** — 5 dropped on 54 queries,
2 on the remaining 2.

### B.7 The cost C2 does carry — context budget

No retrieval metric captures this, so it is measured directly. Characters sent
to the prompt across the 39 Bucket 1 queries:

| Config | Mean chars | Median | Total | vs C0 |
|---|---|---|---|---|
| `C0_baseline` | 2,976 | 2,872 | 116,050 | — |
| `C1_k6` | 3,586 | 3,552 | 139,862 | **+20.5%** |
| `C2_k8` | 4,772 | 4,780 | 186,123 | **+60.4%** |

`C2_k8` buys its recovery with a **60% larger context**. The shipped code states
the intent explicitly — *"FINAL_K stays at 5 so the QA prompt keeps roughly the
same context budget as before the upgrade"* — so this is not a free parameter;
C2 deliberately breaks a stated design constraint.

### B.8 A reproducibility problem found while running this

The first sweep run showed `par-04` improving from rank 3 to rank 1 under C1 —
impossible, since `final_k` cannot reorder. Investigating rather than reporting
it produced a finding that extends RESULTS.md §2.1:

- Within a single process, retrieval is perfectly stable (3 identical passes).
- **Merely opening and querying a store rewrites `chroma.sqlite3` on disk.**
- Across processes, the dense candidate list for some queries differs on the
  **same store directory**. `par-04`'s dense list began with
  `CONTRIBUTING.md` in the `results.json` process and with `quickstart.rst` in
  the first sweep process.

So a "pinned store" is not a fixed object across process invocations. The fix:
**C0 is recomputed in the same process as C1/C2/C3** and that in-process C0 is
the harm baseline, so every config sees an identical dense candidate list and
any difference is attributable to the configuration alone. As a control, the
in-process C0 was compared against the published C0 — **0 mismatching queries
out of 39** — so the baseline in this run is sound and the published Bucket 1
figures are reproduced exactly.

The spurious `par-04` "improvement" is gone from the corrected tables above.

---

## Verdict

**Does any configuration resolve the diagnosed mechanism without a worse
tradeoff elsewhere? Partially, with a cost.**

- **`C1_k6` does nothing.** Zero queries changed. The six-slot window is one
  short of where the interleaving puts dense rank 4. It is not a fix.
- **`C3_gated` is not a fix and is mildly harmful.** It recovers neither loss,
  lowers Hit@1 (48.7 → 46.2), Hit@3 (84.6 → 82.1) and MRR@5 (0.642 → 0.627),
  and trades one improvement for one regression. The rule is sound in principle
  — it removes half the lexical candidates without a magic number — but on this
  corpus the BM25 noise that causes the evictions ranks *above* its own
  per-query median, so the gate does not reach it.
- **`C2_k8` is the only configuration that resolves the mechanism.** It recovers
  both named losses, introduces **zero** new harm in Bucket 1, and leaves the
  Bucket 3 false-positive rate unchanged.

**But `C2_k8` is not a clean win, and three caveats must travel with it:**

1. **The gain is not statistically significant.** Hit@final_k rises 87.2% →
   92.3%, but the intervals [73.3, 94.4] and [79.7, 97.3] overlap heavily. At
   n = 39 this is a 2-query difference.
2. **It rests on the two queries that motivated the sweep.** `crs-03` and
   `crs-06` are the same queries that produced the hypothesis. That is what a
   pre-registered test is for — the prediction was made in advance and the
   mechanism is arithmetic, not curve-fitting — but the evidence is still two
   observations, both in the smallest category (`cross_file`, n = 7).
3. **It costs 60% more context.** It fixes recall by widening the window rather
   than by ranking better, and it breaks a design constraint the shipped code
   states explicitly.

**Nothing here shows hybrid retrieval beating dense-only.** Dense-only scored
89.7% Hit@5 on Bucket 1 (RESULTS.md §3.2); `C2_k8` reaches 92.3% only at
`final_k = 8`, comparing 8 retrieved chunks against dense-only's 5. That is not
a like-for-like comparison and is not offered as one.

**What this data does not support:** changing the shipped default. The effect is
two queries wide, statistically indistinguishable from noise, carries a real
context cost, and was measured on one repository with one embedding model.

---

## Reproducing

```bash
python -m benchmark.run_sweep --store <chroma_bench_dir>
python -m pytest tests/test_benchmark_sweep.py -q
```

Artifact: `benchmark/sweep_results.json`. Tests: `tests/test_benchmark_sweep.py`
(17 tests, synthetic fixtures only — the sweep's measured numbers are
measurements, not invariants, and are deliberately not asserted on).
