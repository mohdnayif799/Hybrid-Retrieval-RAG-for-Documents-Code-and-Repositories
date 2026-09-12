# Blind Quality Audit Log

**Written: 2026-09-09 20:24:18 IST**

This log was written **before** any v2 retrieval was executed. No arm has been
run against the 94-query v2 set at the time of writing. `benchmark/results.json`
at this moment still contains only the v1 (n=56) outcomes.

---

## A disclosure that has to come first

The protocol requires judging each query with **no visibility into any arm's
retrieval results**. For the 38 **new** queries that condition holds exactly:
nothing has been run against them, and I could not know their outcomes if I
wanted to.

For the **original 56 it does not hold, and I will not pretend otherwise.** I
analysed their outcomes in detail in earlier rounds. I know specifically that:

- `crs-03` and `crs-06` were hybrid's two total losses;
- `par-04` is bistable across process launches;
- `con-01` was missed entirely by dense-only;
- `par-08` regressed under the C3 gate configuration.

I cannot un-know that. A "blind" judgment from me on those 56 would be a
fiction, and asserting it would be exactly the kind of unfalsifiable claim this
protocol exists to prevent.

**So the rule I am binding myself to is stricter than the brief's:**

> **No query from the original 56 will be removed, edited, or reweighted in
> this round — regardless of what this audit finds.** Concerns about them are
> *recorded here for the user to act on*, and nothing more.

That makes outcome-motivated removal structurally impossible for the set where
I am contaminated, rather than merely forbidden. Removals are confined to the
38 new queries, where blindness is real.

---

## Criteria

1. **GT-ERROR** — the labelled location is actually wrong.
2. **AMBIGUOUS** — a reasonable reading has a different valid answer not captured.
3. **UNREALISTIC** — no real user would phrase it this way.
4. **DUPLICATE** — redundant information need with another query.
5. **MISLABELLED** — category does not match what the query actually tests.

---

## Part 1 — The 38 new queries (genuinely blind)

### Bucket 1 additions (5)

| id | Judgment |
|---|---|
| `eid-09` `get_netrc_auth` | **PASS.** Defined in `utils.py`; verified. Realistic — users search identifiers verbatim. |
| `eid-10` `guess_json_utf` | **PASS.** Defined in `utils.py`; verified. |
| `lit-09` `DEFAULT_RETRIES` | **PASS.** Constant in `adapters.py`; verified. |
| `par-09` netrc paraphrase | **PASS.** Overlap 0.20. Avoids "netrc"/"auth"/"credentials". GT `utils.py` + `sessions.py` both genuinely involved (`trust_env`). |
| `crs-08` ContentDecodingError | **PASS.** Genuinely two files: exception defined in `exceptions.py`, decoding attempted in `models.py`. Satisfies cross_file. |

### doc_factual (6 new)

| id | Judgment |
|---|---|
| `dfa-06` 12 parts of a formal report | **PASS.** GT @6155-7253 is the enumerated list. |
| `dfa-07` memo headings | **FLAG — DUPLICATE (partial).** GT @1483-3057 is shared with `dco-07` and `dpa-04`. See the cluster note below. |
| `dfa-08` AirPods package contents | **PASS.** GT @50900-51600. Note the source spells it "Air Pods"; the question uses "AirPods", which is *good* — it tests tokenisation, not memorisation. |
| `dfa-09` four report types | **PASS.** GT @3057-4151. |
| `dfa-10` user-manual title page | **FLAG — DUPLICATE (partial)** with `ddi-04` (same GT range @40296-42582). `dfa-10` asks for one element's contents, `ddi-04` for the whole list. Distinguishable but overlapping. |
| `dfa-11` sample progress report header | **PASS.** GT @13197-14260 contains the To/From block. |

### doc_conceptual (5 new)

| id | Judgment |
|---|---|
| `dco-06` progress vs evaluation | **PASS.** GT @18495-19212. Correctly labelled doc_conceptual, not cross_file — it is a within-document comparison. |
| `dco-07` memo vs letter choice | **PASS.** Distinct from `dfa-07`/`dpa-04`: asks *which to choose and why*, not *what it contains*. |
| `dco-08` kinds of viability | **PASS (rewritten).** Original scored 1.00 overlap and was a doc_exact_term question wearing a conceptual label. Rewritten pre-measurement; now 0.22 and targets a different section. |
| `dco-09` why write operating guidance | **PASS (rewritten).** Original 0.60 → 0.09. |
| `dco-10` role of body language | **PASS.** Both copies of the duplicated block listed. **Note: 15 chunks satisfy this GT** — unusually broad, therefore an easier query. Recorded, not removed. |

### doc_paraphrase (6 new)

| id | Judgment |
|---|---|
| `dpa-01` progress | **PASS.** Overlap 0.20. |
| `dpa-02` feasibility | **PASS.** Overlap 0.50 — highest in the category, carried by "document"/"plan". Acceptable. |
| `dpa-03` user manual | **PASS.** Overlap 0.17. |
| `dpa-04` internal note layout | **FLAG — DUPLICATE.** Same GT (@1483-3057) and substantially the same information need as `dfa-07` ("what headings does a memo carry"). For a memo, "how do I lay it out" and "what headings does it carry" resolve to the same answer. |
| `dpa-05` evaluation | **PASS.** Overlap 0.20. |
| `dpa-06` holding attention in a talk | **PASS.** Overlap **0.00** — the cleanest query in the set. |

### doc_exact_term (5 new)

| id | Judgment |
|---|---|
| `dex-01` Kinesics | **PASS.** Both duplicate ranges listed. Broad GT (15 chunks) noted, as with `dco-10`. |
| `dex-02` Air Pods User Manual | **PASS.** |
| `dex-03` Operational Manual | **FLAG — DUPLICATE (partial)** with `ddi-06`; `dex-03`'s range @45540-47730 strictly contains `ddi-06`'s @45917-47730. Different types (lookup vs discrimination), so retained, but they are not independent observations. |
| `dex-04` Executive Summary | **PASS.** Both occurrences listed. |
| `dex-05` Manuscript Format | **PASS.** |

### doc_discrimination (8 new)

| id | Judgment |
|---|---|
| `ddi-01` good feasibility report | **PASS.** GT @28207-28667 only; three siblings must be rejected. Exactly the intended test. |
| `ddi-02` project report characteristics | **PASS.** |
| `ddi-03` general good-report features | **PASS.** Both general sections listed; the project- and feasibility-specific ones deliberately excluded. |
| `ddi-04` user manual elements | **PASS** (see `dfa-10` overlap note). |
| `ddi-05` product manual elements | **PASS.** |
| `ddi-06` operational manual elements | **PASS** (see `dex-03` overlap note). |
| `ddi-07` three manual types compared | **PASS.** GT is the ASCII table @54094-56381, verified to span chunks 68–70. Only 1 chunk minimum matches, making this the most precise GT in the set. |
| `ddi-08` purpose vs structure of a project report | **PASS.** |

### Bucket 3 additions (3 new)

| id | Judgment |
|---|---|
| `una-08` Gantt chart | **PASS.** "gantt" = 0 across all three documents. Genuinely distinct failure mode: adjacent-but-absent within a covered topic (visual aids). |
| `una-09` minutes of a meeting | **PASS.** 0 occurrences. |
| `una-10` resume / CV | **PASS.** Both terms 0 occurrences. |

### Removals from the new set

**One removal**, on criterion DUPLICATE:

- **`dpa-04` — REMOVED.** Same ground-truth range and substantially the same
  information need as `dfa-07`. Keeping both inflates the apparent n of the
  document bucket without adding independent signal. `dfa-07` is retained
  because it is the more concrete phrasing and `doc_factual` is the larger,
  better-established category. This removal is made **before any v2 arm has
  been run**, so it cannot be outcome-motivated — no outcome exists yet.

`dfa-10`/`ddi-04` and `dex-03`/`ddi-06` are **retained** despite partial
overlap: each pair tests a genuinely different capability (specific-fact
lookup vs. sibling discrimination). Their non-independence is recorded here so
it can be discounted when reading category-level results.

---

## Part 2 — The original 56 (NOT blind; recorded only, nothing removed)

Concerns recorded for the user. **No action taken on any of these.**

| id | Concern | Criterion |
|---|---|---|
| `par-04` | Bistable across process launches (2/5 vs 3/5 runs give different dense rank). This is a *measurement reproducibility* problem, **not** one of the five quality criteria — and removing it would conveniently move Bucket 1 Hit@1. Explicitly retained. | none (out of scope) |
| `dco-05` | Bias audit flags 0.60 overlap; reviewed in v1 and kept because the overlap is the entity's own name ("black hole", "star"). Unchanged. | AMBIGUOUS (rejected) |
| `crs-04` | Requires 3 locators, one of which (`certs.py`) is a single chunk out of 1222. Structurally very hard for Recall@5. Not an error — a difficulty property. | none |
| `crs-03`, `crs-06` | I know these were hybrid's total losses. **I have deliberately not audited them further**, because any judgment I reached would be unfalsifiably contaminated by that knowledge. | withheld |
| Bucket 1 GT breadth | **Systemic, affects all 44.** File-level ground truth means a "hit" is any of 51–187 chunks of the right file (median 59–164). Bucket 2's range-level GT means 2–4 specific chunks. These are not comparable difficulty levels. | MISLABELLED (systemic) |

The last row is the most consequential finding in this audit and is carried
into the report's limitations: **Bucket 1 and Bucket 2 Hit@5 must not be
compared to each other.** Bucket 2 scoring lower would not mean document
retrieval is worse; it would mean its ground truth is roughly 30× more precise.

---

## Part 3 — Post-hoc outcome cross-reference

**Written: 2026-09-09 20:32:27 IST** — eight minutes after Parts 1–2, and
after the v2 evaluation had been executed. The ordering is the point: Parts
1–2 above, including the `dpa-04` removal, were fixed before any v2 arm ran.

### The uncomfortable coincidence, stated plainly

`dpa-04` was removed on DUPLICATE grounds before anything was run. Scoring it
post-hoc purely for disclosure:

| Arm | `dpa-04` (removed) would have scored |
|---|---|
| `dense_only` | rank 1 — **hit** |
| `bm25_only` | **miss** |
| `hybrid` | rank 3 — hit |

**So the removal happened to delete a query that `bm25_only` failed.** Keeping
it would have made `bm25_only`'s Bucket 2 Hit@5 34/40 = 85.0% instead of
34/39 = 87.2% — the removal flattered `bm25_only` by 2.2 points.

I did not know that when I removed it, and the timestamps show the ordering.
But "I didn't know" is exactly what someone who *had* peeked would also say,
so the disclosure matters more than the assurance. Three things make it
checkable rather than merely asserted:

1. The removal criterion (DUPLICATE) is verifiable independently of any
   outcome: `dpa-04` and `dfa-07` share the identical ground-truth range
   @1483-3057.
2. The **retained** twin `dfa-07` scores a hit on **all three arms**
   (rank 1/1/1), so the pair was not selected to favour anything — if I had
   been optimising for an arm I would have kept the discriminating one.
3. Reinstating `dpa-04` changes `bm25_only` Bucket 2 from 87.2% to 85.0%,
   which does not alter any conclusion in the report: hybrid and BM25 still
   lead dense-only, and Bucket 2's McNemar is still short of significance.

**If you would rather have the query back, reinstate it** — the reasoning
above is recorded so you can overrule it. I have left it removed because the
duplicate-information-need argument stands on its own.

### The original 56

No query from the original 56 was removed, edited, or reweighted, as
committed in the disclosure at the top. `crs-03`, `crs-06`, `par-04`,
`con-01` and `crs-04` all survive into v2 unchanged, including the two that
hybrid loses and the one whose instability moves Bucket 1 Hit@1.

### Did the audit's flags correlate with failure?

Of the queries flagged FLAG-but-retained in Part 1:

| id | Flag | v2 outcome (hybrid) |
|---|---|---|
| `dfa-07` | DUPLICATE (partial) | hit, rank 1 |
| `dfa-10` | DUPLICATE (partial) | hit, rank 1 |
| `dex-03` | DUPLICATE (partial) | hit, rank 1 |
| `dco-10` | broad GT (15 chunks) | hit, rank 1 |
| `dex-01` | broad GT (15 chunks) | hit, rank 1 |

All five flagged queries **succeeded**. The flags identified redundancy and
over-broad ground truth, not difficulty — which is what a quality audit
should find, and is evidence the criteria were not a proxy for "queries an
arm got wrong".

The genuinely hard category, `doc_paraphrase` (20% across all three arms),
was **not** flagged by the audit at all. Its difficulty was invisible to a
blind quality review, which is the correct outcome: it is a real retrieval
failure, not a defect in the questions. Section-level inspection confirms it —
for `dpa-02` and `dpa-03` all five retrieved chunks come from the correct
document but the wrong sections.
