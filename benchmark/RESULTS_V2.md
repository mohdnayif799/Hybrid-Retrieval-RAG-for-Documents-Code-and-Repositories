# Retrieval Evaluation v2 — Expanded Corpus and Query Set

Supersedes the measurements in [RESULTS.md](RESULTS.md) (v1, n=56), which is
preserved verbatim as [RESULTS_v1_n56.md](RESULTS_v1_n56.md) with its raw data
in `results_v1_n56.json`. Configuration work from v1 is in
[TUNING.md](TUNING.md); the audit trail is in
[blind_audit_log.md](blind_audit_log.md).

**No shipped file was changed.** SHA-256 prefixes verified unchanged for
`src/hybrid_retrieval.py` (`e889e6bb…`), `app.py` (`94389537…`),
`src/vector_store.py` (`53139d67…`), `src/citations.py` (`59e3e5bc…`),
`src/chunking.py` (`e713ae8b…`). **No Gemini API call was made.**

---

## 1. What changed

### 1.1 Corpus

| | v1 | v2 |
|---|---|---|
| Repo (pin unchanged) | 1111 chunks | 1111 chunks |
| `Deep_Learning_Unit1_Quick_Notes.pdf` | 4 | 4 |
| `space_planets_moons_advanced.txt` | 8 | 8 |
| **`V_SEM_3_to_5_units_ETCE.docx`** | — | **99** |
| **Total** | 1123 | **1222** |

New document: 122,879 bytes, `sha256 ad6579aab7f10f54…`, 78,397 extracted
characters, ingested through the existing `data_ingestion → chunk_documents`
path with no new ingestion code. Repo pin `dae7ef63…` unchanged.

### 1.2 Three structural facts about the .docx, all verified

**(a) File-level ground truth is unusable for it.** `Docx2txtLoader` returns
**one** Document at `page 0`, so all 99 chunks share a single `(file, page)`
locator. A file-level "hit" would be satisfied by any chunk of a
78,000-character document. Ground truth for this document is therefore
**`start_index` ranges**, matched by interval overlap
(`benchmark/metrics.chunk_matches`).

**(b) A 5,439-character block is duplicated verbatim** — `[62169, 67608)` and
`[67648, 73087)` — covering all the kinesics / non-verbal / presentation
material. Both ranges are listed for every question answered there; listing
one would score a correct retrieval of the other copy as a miss.

**(c) Tables do not survive ingestion intact.** The Progress-vs-Evaluation
comparison is a real Word table that `Docx2txtLoader` flattens into
alternating paragraphs (row pairing survives by adjacency; row *structure*
does not). A separate ASCII-art table @54241-56282 is **split across three
chunks** (68, 69, 70), so no single chunk holds the whole comparison. Both are
reflected in the ground truth rather than worked around.

### 1.3 Query set: 56 → 93

| Bucket | v1 | v2 | Change |
|---|---|---|---|
| 1 — code | 39 | **44** | +5 |
| 2 — documents | 10 | **39** | +30, −1 removed |
| 3 — unanswerable | 7 | **10** | +3 |
| **Total** | 56 | **93** | |

Bucket 2 sub-categories, with three new tags so the harder capabilities are
reported separately rather than blended into the old easier ones:

`doc_factual` 5→11 · `doc_conceptual` 5→10 · `doc_paraphrase` 0→5 ·
`doc_exact_term` 0→5 · `doc_discrimination` 0→8

`cross_file` remains **code-only** (7→8). The Progress-vs-Evaluation question
is authored as `doc_conceptual`, not `cross_file` — it is a within-document
comparison, and labelling it cross_file would have been a category error.

### 1.4 Every change to the query set, with its criterion

| Change | Criterion | Basis |
|---|---|---|
| **`dpa-04` removed** | DUPLICATE | Identical GT range @1483-3057 and substantially the same information need as `dfa-07`. Removed **before any v2 arm ran**. |
| **`dco-08` rewritten** | MISLABELLED | Original scored **1.00** bias-audit overlap; its only content tokens were "feasibility" and "report" — a `doc_exact_term` question wearing a conceptual label. Now 0.22, and retargeted to a different section to avoid a near-duplicate with `dpa-02`. |
| **`dco-09` rewritten** | MISLABELLED | 0.60 → **0.09**; original reused the section heading's own words. |
| **38 queries added** | — | Authored question-first; all locators verified against the freshly built corpus. |
| **0 of the original 56 removed or edited** | — | See §1.5. |

### 1.5 Why nothing was removed from the original 56

The protocol requires blind judgment. **For the original 56 I am not blind and
did not claim to be** — I analysed their outcomes in earlier rounds and know
specifically that `crs-03`/`crs-06` were hybrid's losses, `par-04` is
bistable, and `con-01` was missed by dense.

So I bound myself to a stricter rule than the brief's: **no query from the
original 56 was removed, edited or reweighted, whatever the audit found.**
Concerns are recorded in the audit log for you to act on. That makes
outcome-motivated removal structurally impossible where I am contaminated,
rather than merely forbidden. Removals were confined to the 38 new queries,
where blindness is real because nothing had been run against them.

**Disclosed coincidence:** the one removal (`dpa-04`) happened to delete a
query that `bm25_only` fails — flattering `bm25_only` by 2.2 points in Bucket
2 (87.2% vs 85.0% if reinstated). Full disclosure, including why it does not
change any conclusion and how to reinstate it, is in
[blind_audit_log.md](blind_audit_log.md) Part 3.

---

## 2. Verification before measuring

### 2.1 Determinism, re-checked on the new composition

Not assumed to carry over. Two fresh builds, 93-query set:

| Arm | Identical | Differing |
|---|---|---|
| `bm25_only` | **56/56** | — |
| `hybrid` | 54/56 | `par-04`, `con-08` |
| `dense_only` | 53/56 | `par-04`, `con-08`, `una-07` |

Same qualitative pattern as v1: BM25 perfectly stable, dense/hybrid unstable
on a handful. Full-report diff across both builds: Bucket 2, provenance and
Bucket 3 **identical**; the only mover is **`par-04`**, exactly as in v1.

**`par-04` is bistable.** Across five separate process launches on one store:
2/5 give dense rank 1, 3/5 give dense rank 2. It moves Bucket 1 Hit@1 by one
query (**53.8% ↔ 56.4%**) and MRR@5 by ≈0.013.

**It was deliberately retained.** Measurement instability is not one of the
five audit criteria, and removing it would conveniently move a headline
number. Bucket 1 Hit@1 and MRR@5 should be read as carrying a ±1-query
reproducibility band.

### 2.2 Bias audit (project tokenizer)

| Category | n | median | mean |
|---|---|---|---|
| `exact_identifier` | 10 | 1.00 | **1.00** |
| `doc_exact_term` | 5 | 1.00 | **1.00** |
| `literal_string` | 9 | 1.00 | 0.96 |
| `doc_factual` | 11 | 0.60 | 0.63 |
| `doc_discrimination` | 8 | 0.60 | 0.52 |
| `doc_conceptual` | 10 | 0.50 | 0.41 |
| `cross_file` | 8 | 0.40 | 0.35 |
| `paraphrase` | 9 | 0.23 | 0.29 |
| `conceptual` | 8 | 0.29 | 0.29 |
| **`doc_paraphrase`** | 5 | **0.20** | **0.20** |

High overlap for the exact-term categories is correct by construction.
`doc_paraphrase` at 0.20 (one query at **0.00**) is the cleanest in the set.

`doc_discrimination` at 0.52 is **expected, not contamination**: a
discrimination question must name its target ("feasibility report", "product
manual") to specify which sibling it means. That is the entity's name, not
inherited phrasing — the same reasoning applied to `dco-05` in v1.

### 2.3 A ground-truth asymmetry that governs how to read everything below

Measured breadth — how many chunks satisfy each query's ground truth:

| Bucket | Category | min | median | max |
|---|---|---|---|---|
| 1 | `paraphrase` | 59 | **164** | 175 |
| 1 | `literal_string` | 51 | 72 | 187 |
| 1 | `exact_identifier` | 7 | 59 | 71 |
| 2 | `doc_discrimination` | **1** | **4** | 5 |
| 2 | `doc_factual` | 1 | 3 | 8 |

**Bucket 1 and Bucket 2 Hit@5 are not comparable.** A Bucket 1 hit means "any
of 51–187 chunks of the right file"; a Bucket 2 hit means "one of 2–4 specific
chunks". Bucket 2's ground truth is roughly **30× more precise**. Bucket 2
scoring lower would not mean document retrieval is worse.

---

## 3. Bucket 1 — Code corpus (n = 44)

### 3.1 Per category (Hit@5, 95% Wilson CI)

| Category | n | dense_only | bm25_only | hybrid |
|---|---|---|---|---|
| exact_identifier | 10 | 90.0% [59.6, 98.2] | 90.0% [59.6, 98.2] | 90.0% [59.6, 98.2] |
| paraphrase | 9 | 77.8% [45.3, 93.7] | 66.7% [35.4, 87.9] | 77.8% [45.3, 93.7] |
| conceptual | 8 | 87.5% [52.9, 97.8] | 62.5% [30.6, 86.3] | **100.0% [67.6, 100.0]** |
| literal_string | 9 | 100.0% [70.1, 100.0] | 100.0% [70.1, 100.0] | 100.0% [70.1, 100.0] |
| cross_file | 8 | **87.5% [52.9, 97.8]** | 50.0% [21.5, 78.5] | 62.5% [30.6, 86.3] |

### 3.2 Aggregate

| Metric | dense_only | bm25_only | hybrid |
|---|---|---|---|
| Hit@1 | **56.8% [42.2, 70.3]** | 36.4% [23.8, 51.1] | 52.3% [37.9, 66.2] |
| Hit@3 | 79.5% [65.5, 88.8] | 63.6% [48.9, 76.2] | **84.1% [70.6, 92.1]** |
| Hit@5 | **88.6% [76.0, 95.0]** | 75.0% [60.6, 85.4] | 86.4% [73.3, 93.6] |
| MRR@5 | **0.692** | 0.505 | 0.664 |
| span Hit@5 (n=16) | 68.8% [44.4, 85.8] | 56.2% [33.2, 76.9] | **75.0% [50.5, 89.8]** |
| cross_file Recall@5 | **0.521** | 0.250 | 0.458 |

### 3.3 Significance — both tests

| Test | v1 (n=39) | v2 (n=44) |
|---|---|---|
| McNemar, Hit@5 flips | 3 discordant, **p = 1.0000** | 3 discordant, **p = 1.0000** |
| Sign test, rank changes | 13w/8b, n=21, **p = 0.3833** | 13w/9b, n=22, **p = 0.5235** |

Both agree with `scipy.stats.binomtest` to 1e-12. **Still underpowered after
expansion.** Only 3 discordant pairs — fewer than the 6 required for any
arrangement to reach p<0.05. The v1 conclusion is unchanged and did not
strengthen: **this experiment still cannot distinguish dense from hybrid on
the code corpus.**

### 3.4 Provenance (hybrid's 38 Bucket 1 successes)

| Origin | Count | Share |
|---|---|---|
| `vector+bm25` | 27 | 71% |
| `vector` only | 7 | 18% |
| `bm25` only | **4** | **11%** |

`literal_string` is **9/9 `vector+bm25`** — BM25 is still the sole discoverer
in nothing, in the category it was expected to own. This replicates v1 (4/34 →
4/38).

### 3.5 Harm analysis — 9 better, 13 worse, 22 unchanged

`crs-03` and `crs-06` are **still lost entirely** (dense rank 4 → hybrid miss).
The §3.6 interleaving mechanism from RESULTS.md replicates unchanged on the
expanded set. New in v2: `eid-10` (`guess_json_utf`) joins the regression list
at dense 1 → hybrid 2; `lit-09` (`DEFAULT_RETRIES`) joins the improvements at
dense 2 → hybrid 1.

---

## 4. Bucket 2 — Documents (n = 39). **No longer a ceiling.**

v1 labelled this "ceiling effect expected — not used for the comparison". That
label no longer applies, and part of the reason is methodological, not just
corpus size.

### 4.1 Per category (Hit@5)

| Category | n | dense_only | bm25_only | hybrid |
|---|---|---|---|---|
| doc_factual | 11 | 72.7% [43.4, 90.3] | **100.0% [74.1, 100.0]** | **100.0% [74.1, 100.0]** |
| doc_conceptual | 10 | 90.0% [59.6, 98.2] | 90.0% [59.6, 98.2] | **100.0% [72.2, 100.0]** |
| **doc_paraphrase** | 5 | **20.0% [3.6, 62.4]** | **20.0% [3.6, 62.4]** | **20.0% [3.6, 62.4]** |
| doc_exact_term | 5 | 80.0% [37.6, 96.4] | 100.0% [56.6, 100.0] | 80.0% [37.6, 96.4] |
| doc_discrimination | 8 | 100.0% [67.6, 100.0] | 100.0% [67.6, 100.0] | 100.0% [67.6, 100.0] |

### 4.2 Aggregate

| Metric | dense_only | bm25_only | hybrid |
|---|---|---|---|
| Hit@1 | 66.7% [51.0, 79.4] | 66.7% [51.0, 79.4] | **69.2% [53.6, 81.4]** |
| Hit@3 | 74.4% [58.9, 85.4] | **84.6% [70.3, 92.8]** | **84.6% [70.3, 92.8]** |
| Hit@5 | 76.9% [61.7, 87.4] | **87.2% [73.3, 94.4]** | **87.2% [73.3, 94.4]** |
| MRR@5 | 0.706 | 0.740 | **0.762** |

**McNemar: dense 0 wins, hybrid 4 wins, 4 discordant, p = 0.1250.** Every
discordant pair favours hybrid — but 4 < 6, so this is still short of
significance. It is the first sign in this project of hybrid helping, and it
is *suggestive, not established*.

### 4.3 The v1 ceiling was substantially a measurement artifact

Bucket 2 Hit@5 computed both ways on identical retrievals:

| Arm | file-level GT (v1 style) | range-level GT (v2) |
|---|---|---|
| `dense_only` | 97.4% | **76.9%** |
| `bm25_only` | 97.4% | **87.2%** |
| `hybrid` | 100.0% | **87.2%** |

File-level ground truth reports a near-ceiling on the *same* retrievals that
range-level scoring shows failing 5–9 times out of 39. v1's "ceiling effect"
conclusion was therefore partly an artifact of coarse ground truth, not solely
a consequence of a 12-chunk corpus.

### 4.4 Two findings that contradicted my expectations

**`doc_discrimination` was supposed to be the hard category. It scored 100%
on all three arms.** The near-duplicate section clusters — four
"Characteristics of…" sections, three manual-element lists — did not defeat
retrieval at all. My hypothesis that topical adjacency would cause sibling
confusion was **wrong on this corpus**, and I am reporting it as wrong rather
than reframing it.

**`doc_paraphrase` is the genuinely hard category at 20% for every arm** — and
it was *not* flagged by the blind audit. Section-level inspection confirms
this is real retrieval failure, not a ground-truth defect: for `dpa-02` and
`dpa-03`, **all five retrieved chunks come from the correct document but the
wrong sections**. Under file-level ground truth every one of those would have
counted as a hit. When a question shares no vocabulary with its answer,
retrieval finds the right *document* and the wrong *part of it* — and only
range-level ground truth can see that.

---

## 5. Bucket 3 — Unanswerable (n = 10)

| Arm | v1 (n=7) | v2 (n=10) |
|---|---|---|
| `dense_only` | 0/7 = 0.0% [0, 35] | 0/10 = **0.0% [0, 28]** |
| `bm25_only` | 1/7 = 14.3% [3, 51] | 1/10 = 10.0% [2, 40] |
| `hybrid` | 2/7 = 28.6% [8, 64] | 1/10 = **10.0% [2, 40]** |

Three additions test a **distinct failure mode** from v1's: v1's unanswerables
were off-domain (HTTP/3, Proxima Centauri); v2 adds *adjacent-but-absent*
topics — a Gantt chart in a document that covers bar charts, tree diagrams and
flowcharts; meeting minutes; a résumé. All verified 0 occurrences.

All three arms still return 5 chunks for every unanswerable question. **Nothing
in the retrieval layer abstains.** This measures a refusal *precondition* only;
whether the generator refuses needs the API and was not run.

---

## 6. Phase 6 — Testing a pre-existing hypothesis on new data

**A correction first.** The brief describes 0.55/0.45 and 0.60/0.40 as
"already-published, pre-registered". **They were not.** TUNING.md's
pre-registered set was C0/C1_k6/C2_k8/C3_gated; RESULTS.md §9 listed "lower
`BM25_WEIGHT`" as one of three unexplored follow-ups without specifying
values. They are pre-registered **in this round** and labelled as such in
`benchmark/sweep.py`.

**Prediction, recorded before running:** at 0.55/0.45, dense rank *r* scores
`0.55/(60+r)` and BM25 rank *r* scores `0.45/(60+r)`. Dense rank 10
(0.007857) still beats BM25 rank 1 (0.007377), so **no BM25-only chunk can
enter the top 5**. The reweight should therefore *eliminate* BM25's unique
contribution, not preserve it.

### 6.1 Results

| Config | B1 Hit@5 | B1 Hit@final_k | B1 MRR | B2 Hit@5 | B2 MRR |
|---|---|---|---|---|---|
| `C0_baseline` (0.50/0.50) | 86.4% [73.3, 93.6] | 86.4% | 0.664 | 87.2% [73.3, 94.4] | 0.774 |
| `W1_55_45` | **90.9% [78.8, 96.4]** | 90.9% | 0.684 | 84.6% [70.3, 92.8] | 0.766 |
| `W2_60_40` | **90.9% [78.8, 96.4]** | 90.9% | 0.696 | 82.1% [67.3, 91.0] | 0.761 |
| `C3_gated` | 86.4% | 86.4% | 0.649 | **92.3% [79.7, 97.3]** | **0.802** |
| `C1_k6` | 86.4% | 86.4% | 0.664 | 87.2% | 0.774 |
| `C2_k8` | 86.4% | **93.2% [81.8, 97.7]** | 0.664 | 87.2% | 0.774 |
| `D0_dense_k5` | 88.6% [76.0, 95.0] | 88.6% | 0.692 | 79.5% [64.5, 89.2] | 0.732 |
| `D1_dense_k8` | 88.6% | **93.2% [81.8, 97.7]** | 0.692 | 79.5% | 0.732 |

### 6.2 The prediction held exactly

| Config | `vector+bm25` | `vector` | **`bm25` only** |
|---|---|---|---|
| `C0_baseline` | 27 | 7 | **4** |
| `W1_55_45` | 27 | 13 | **0** |
| `W2_60_40` | 27 | 13 | **0** |

**Reweighting does not rebalance BM25's contribution — it deletes it.** The
four queries BM25 alone rescued under C0 become `vector` finds; no BM25-only
chunk reaches the top 5 at either weighting. The hypothesis as stated in the
brief — "recover the ranking distortion *without materially reducing BM25's
ability to promote genuine unique finds*" — is **falsified**. It recovers the
distortion precisely *by* removing those finds.

### 6.3 The equal-budget comparison erases the v1 C2_k8 result

The gap v1 left open: hybrid@8 was only ever compared against dense@5.

| | Hit@final_k (Bucket 1) |
|---|---|
| `C2_k8` (hybrid, k=8) | **93.2% [81.8, 97.7]** |
| `D1_dense_k8` (dense, k=8) | **93.2% [81.8, 97.7]** |

**Identical.** At equal budget, hybrid@8 offers no advantage over dense@8 on
the code corpus. TUNING.md's "C2_k8 recovers both named losses with zero new
harm" was real, but the recovery came from *more slots*, not from fusion —
dense-only recovers `crs-03` and `crs-06` at rank 4 without any fusion at all.

### 6.4 Verdict

**The reweight hypothesis does not replicate as stated, and one of its two
claims is falsified.** It improves Bucket 1 (86.4% → 90.9%) but:

- it does so by suppressing BM25 entirely (bm25-only provenance 4 → 0);
- it **degrades Bucket 2** (87.2% → 84.6% → 82.1%), the bucket where hybrid
  showed its only sign of genuine benefit;
- neither the Bucket 1 gain nor the Bucket 2 loss is statistically
  significant — the intervals overlap heavily throughout;
- and at equal budget, plain dense@5 already scores 88.6% on Bucket 1, within
  the reweight's interval.

The one configuration that improves Bucket 2 is `C3_gated` (87.2% → 92.3%) —
the config v1 concluded was useless on the code corpus. That is interesting and
**not** something I went looking for, but it is 2 queries on n=39 with
overlapping intervals, and it is reported as a lead, not a result.

---

## 7. v1 → v2: did conclusions change?

| Question | v1 | v2 | Verdict |
|---|---|---|---|
| Does hybrid beat dense on code? | No; p=1.0, 3 discordant | No; p=1.0, 3 discordant | **Unchanged, still underpowered** |
| Is BM25 the sole discoverer often? | 4/34 (12%) | 4/38 (11%) | **Replicated** |
| Does equal-weight RRF evict dense 4–5? | Yes; `crs-03`/`crs-06` lost | Yes; still lost | **Replicated** |
| Is Bucket 2 a ceiling? | "Yes, skip it" | **No** — 76.9–87.2% | **Reversed** |
| Does hybrid help anywhere? | No evidence | Bucket 2: 4/4 discordant favour hybrid, p=0.125 | **New, suggestive only** |
| Does `final_k=8` help? | Appeared to | **No** at equal budget | **Reversed** |
| Are near-duplicate sections hard? | untested | **No** — 100% all arms | **Hypothesis wrong** |

**The headline comparison is still underpowered after expansion.** Bucket 1
grew 39→44 and produced the *same* 3 discordant pairs. Expanding the query set
did not resolve the dense-vs-hybrid question; it would need either far more
queries or a corpus where the arms genuinely diverge.

---

## 8. Citation verification

| Check | Result |
|---|---|
| Repo `start_line` | **1111/1111**, 0 mismatched |
| PDF page content | **4/4**, 0 mismatched |
| Text offset | **8/8**, 0 mismatched |
| **`.docx` `start_index`** | **99/99**, 0 mismatched (78,397 chars) |
| Citation group collisions | **0** across 98 groups |
| PPTX slide verification | **not measured** — no PPTX in corpus |

**1222 objective checks, zero failures.** The `.docx` check is new and matters:
the document half's ground truth is expressed in `start_index` ranges, so a
wrong `start_index` would silently score the wrong regions. Grouping is
genuinely exercised — 6 basenames are shared by multiple paths (`Makefile` ×8,
`README.md` ×4, `LICENSE` ×3).

---

## 9. Tests

| | |
|---|---|
| Before this round | **207 passed** |
| `tests/test_benchmark_specs.py` (new) | +28 |
| `tests/test_benchmark_sweep.py` (extended) | +5 |
| **After** | **240 passed**, 58 subtests |

Synthetic fixtures only; no test asserts on a measured benchmark number. One
pre-existing test failed correctly when the config set grew from 4 to 6 and was
updated to pin the v2 set.

---

## 10. Limitations

- **One code repository** (`psf/requests`), one embedding model
  (`all-MiniLM-L6-v2`), one chunking configuration. Nothing here transfers
  automatically to another repo, language or embedder.
- **Documents: one substantial real document plus one 2-page PDF and one 7 KB
  text file.** The .docx now supplies 99 of 111 document chunks, so Bucket 2 is
  effectively a single-document benchmark.
- **That document is unusual** — it contains a 5,439-character verbatim
  duplicate and four near-duplicate section clusters. Convenient for
  discrimination testing; not representative of ordinary prose.
- **Bucket 1 and Bucket 2 ground truth differ ~30× in precision.** Their
  numbers must not be compared to each other.
- **Still underpowered where it matters most.** Bucket 1: 3 discordant pairs.
  Bucket 2: 4. Both below the 6 needed for significance.
- **`par-04` is not reproducible.** Bucket 1 Hit@1 carries a ±1-query band
  (53.8% ↔ 56.4%) depending on process launch.
- **Queries authored and blind-judged by the same agent**, even under the
  logged protocol. That cannot rule out systematic blind spots shared between
  authoring and auditing — a question I would not think to ask is absent from
  both. It also cannot rule out unconscious calibration of difficulty. The
  logged ordering and timestamps constrain *outcome-motivated* editing; they do
  not make the author impartial.
- **For the original 56 I was not blind at all**, which is why none were
  removed. Their quality is therefore *asserted from v1*, not re-established.
- **Retrieval only.** No answer was generated; no API was called. Nothing here
  establishes that any configuration produces better answers.

---

## 11. What cannot be concluded

- **Not** that hybrid beats dense on code retrieval. p = 1.0 (McNemar) and
  p = 0.5235 (sign test) at n=44.
- **Not** that hybrid beats dense on documents. 4/4 discordant pairs favour
  hybrid, but p = 0.125.
- **Not** that reweighting RRF is an improvement. It falsified its own stated
  rationale and degrades the bucket where hybrid showed promise.
- **Not** that `final_k=8` helps. At equal budget it matches dense@8 exactly.
- **Not** that near-duplicate sections are hard for retrieval — measured at
  100% for all arms, contradicting the hypothesis that motivated the category.
- **Not** that `doc_paraphrase`'s 20% generalises. n=5, CI [3.6, 62.4].
- **Not** that any arm would cause a model to hallucinate or refuse.

### What the evidence does support

1. **Ground-truth granularity changes conclusions.** The same retrievals score
   97.4–100% at file level and 76.9–87.2% at range level. v1's ceiling was
   partly an artifact of coarse ground truth.
2. **Vocabulary mismatch is the real failure mode.** `doc_paraphrase` fails at
   80% for every arm, retrieving the right document and the wrong section.
   This is where retrieval work would actually pay off.
3. **The RRF interleaving mechanism is arithmetic and replicates**, and the
   reweight's effect on it was predicted correctly in advance — including the
   part that falsified the hypothesis.
