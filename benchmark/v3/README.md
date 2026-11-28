# Retrieval benchmark v3

v3 measures **the retrieval path the app actually runs**, against **span-level
ground truth**, **deterministically**, with **paired confidence intervals**. It
replaces `evaluate_retrieval.py` (v1/v2) as the baseline for retrieval changes.
No Gemini call is made: retrieval happens before generation.

> **Status: Phase 1+2 scored.** The query set was reviewed and frozen
> (`manifest.json`, 127 dev / 54 holdout) before anything was scored. The
> Phase-0 baseline is in [RESULTS_phase0.md](RESULTS_phase0.md). The current
> pipeline (candidate pool + reranker) against that baseline is in
> [RESULTS.md](RESULTS.md); configuration was chosen on dev, and holdout was
> scored once. Why each choice was made: [docs/DECISIONS.md](../../docs/DECISIONS.md).

## Why v3

A diagnosis of the v2 evaluation (November 2026) found four problems that made
its "no significant difference" result uninformative:

| Problem in v2 | v3 answer |
|---|---|
| `benchmark/results.json` predates the README pin: no stored ranking carries `readme_boost`, and the shipped retriever scores 71/83, not 72/83 | The `production` arm calls `build_hybrid_retriever` exactly as `src/rag_chain.py` does, so whatever ships is measured |
| Code gold was file-level: a median of 80 chunks counted as a hit, so code Hit@5 sat near its ceiling | Span-level gold for every query: a median of 2 gold chunks per code query |
| Dense rankings changed between builds and between processes (HNSW) | The eval store uses exhaustive HNSW settings: identical production output across two independent builds for 181/181 questions |
| 83 answerable queries, so only differences of about 15 points were detectable | 162 answerable queries, a frozen dev/holdout split, paired bootstrap intervals, Recall@20 reported separately from ranking quality |

## Files

| File | What it is |
|---|---|
| `queries.yaml` | The frozen query set: v2's 93 queries plus 88 new ones (review edits recorded in each query's notes) |
| `split.json`, `manifest.json` | The stratified 70/30 dev/holdout split and the SHA-256 freeze manifest |
| `REVIEW.md` | Generated review sheet: every new query and every new label, with the exact text each span covers |
| `results.json`, `RESULTS.md` | Current pipeline vs baseline: raw per-query results and the generated report |
| `results_phase0.json`, `RESULTS_phase0.md` | The Phase-0 baseline, kept unchanged |
| `baseline_v3.py` | Frozen copy of the pre-Phase-1 retriever (the `baseline_v3` arm) |
| `corpus.py` | Loads the v2 corpus through the real ingestion and chunking code; verifies the repository by a content digest |
| `gold.py` | Span gold: anchors, evidence checks, parts, near-duplicates, rank metrics |
| `dataset.py` | Loads and validates `queries.yaml` (`python -m benchmark.v3.dataset`) |
| `review.py` | Regenerates `REVIEW.md` (`python -m benchmark.v3.review`) |
| `stats.py` | Paired percentile bootstrap |
| `evaluate.py` | The harness (`python -m benchmark.v3.evaluate`) |
| `freeze.py` | Split and freeze (`python -m benchmark.v3.freeze --verify`) |
| `report.py` | Regenerates `RESULTS.md` from `results.json` |

## Ground truth

Each answerable query has `gold`: spans inside the real parent files. A span is
located by a verbatim **anchor** at load time, never by typed offsets, so it
cannot drift silently. Each span on a new query carries **evidence** (the answer
itself), which must occur inside the span. That check is what makes a label
mean "the answer is here" rather than "a related heading is here".

Spans that share a **part** are alternatives (the same fact in the docs and in
the code); different parts are different pieces of a multi-part answer.

* **Hit@1 / Hit@5 / MRR@5**: is any part found in the top 1 / top 5 (ranking quality).
* **Recall@20**: share of parts found in the top 20 (candidate quality). For
  the production arm this is the candidate pool's top 20, the reranker's input.
* **File-level Hit@5** for code queries: the v2-comparable secondary metric.
* **all-parts@5** (multi-part queries): every part has a chunk in the top 5.
* **Chunk coverage @5 / @20** (list_all queries): share of the gold-span chunks retrieved.
* **gold4plus**: Hit@5 reported separately for queries whose gold covers 4+ chunks.
* **Top-1 signal** (every query, every arm): query-vs-top-1 cosine with the production
  embeddings, plus the arm's own score. Unanswerable queries are excluded from Hit/MRR;
  these signals are recorded for a later refusal threshold.
* **Typo twins**: misspelled queries and their correctly spelled twins on the same span.

A chunk that contains at least 80% of a gold chunk's 5-word shingles also counts
as gold. That covers the .docx's duplicated block and docstrings repeated on two
methods. Symmetric containment and embedding cosine were tried and rejected,
because each admitted chunks that do not contain the answer (details in `gold.py`).

## Arms

| Arm | What it is |
|---|---|
| `production` | `build_hybrid_retriever(store).invoke(q)`: what the app returns (candidate pool -> reranker -> top 5). The harness refuses to run if the reranker did not load |
| `baseline_v3` | the pre-Phase-1 pipeline (MMR + BM25, RRF, README pin) from the frozen `baseline_v3.py`; it reproduces the Phase-0 production top-5 lists exactly |
| `production_no_rerank` | the candidate pool's fused order: what production returns if the reranker fails or runs over budget |
| `dense_plain` | plain similarity search on the same store |
| `bm25` | production's BM25 index |

All arms run in one process on one store. Every aggregate is reported per
split, domain (code / docs) and query type, with paired differences against
`baseline_v3`. Production's per-query log line (fallback, rerank time) is
recorded with its scores. `--split dev` retrieves and scores the dev split
only, for tuning without touching holdout.

## Running

```bash
python -m benchmark.v3.dataset              # validate every span against the corpus
python -m benchmark.v3.review               # regenerate REVIEW.md
python -m benchmark.v3.freeze --verify      # check the frozen files are unchanged
python -m benchmark.v3.evaluate --fidelity  # score (refuses if a frozen file changed)
python -m benchmark.v3.report               # regenerate RESULTS.md
```

`--archive PATH` points any of them at a local `psf/requests` tarball of the
pinned commit, for machines that cannot reach codeload.github.com. The archive
is accepted if its indexable files match `corpus.EXPECTED_CONTENT_DIGEST`.

`--fidelity` also builds a store with Chroma's default HNSW settings (what
production uses) and reports how many production top-5 lists differ from the
exhaustive store. Default HNSW is itself nondeterministic across builds: 9 of
181 lists differed in one pre-freeze build, 5 of 181 in the baseline run.

## Open items

1. **Done:** freeze (stratified 70/30 split, seed 20261123, SHA-256 manifest over
   `queries.yaml`, `split.json` and `gold.py`; the harness and a CI test refuse to
   proceed if any of them change) and the baseline ([RESULTS.md](RESULTS.md)).
   `results.json` records hashes of the retrieval code, the frozen files and the
   harness, and a CI test fails when they no longer match, so a retrieval change
   must ship with re-run results.
2. **Tuning discipline:** decisions are made on `dev`; `holdout` is scored to
   confirm a change once, not to choose between variants. Phase 1+2 followed
   this: Part A experiments were dev-only, and holdout was scored once, in the
   run behind RESULTS.md.
3. **Multi-turn query rewriting is not covered.** After the first message of a
   chat, production does not retrieve with the user's words. `src/rag_chain.py`
   first asks Gemini to rewrite the follow-up into a standalone query, and
   retrieves with that. Every v3 query is a first-turn question, so the
   rewrite step and its effect on retrieval are unmeasured. Covering it needs
   multi-turn conversations as test data and a Gemini API key to run the
   rewriter (or recorded rewrites). Not built in Phase 0.
4. **Production's default HNSW.** The eval scores an exhaustive store; production
   uses Chroma's defaults, which change 5-9 of 181 production top-5 lists
   relative to the exhaustive store, varying by build (4 of 181 in the Phase 1+2
   run). If those are ever judged material, the fix belongs in production
   (collection settings), not in the eval.
5. **Repository identity.** `rid-01` ("What is this repository?") and `rid-04`
   ("What features does this project offer?") miss under the old pipeline and
   the new one, with or without the README pin; holdout's `rid-05` (author and
   license, two parts) misses too. The README's opening is not retrieved for
   identity questions phrased without its vocabulary.
6. **Misspelled code queries.** Still 0 of 4 at Hit@1 on dev (`msp-02`,
   `msp-06` are not retrieved at all; `msp-05`, `msp-11` reach rank 2).
   Fixing them needs query rewriting or spelling correction before retrieval,
   which needs the Gemini key (see item 3).
7. **Code multi-hop.** Dev Hit@5 63.6% (baseline 45.5%), but all-parts@5 for
   multi-part queries is still 20% on dev: the top 5 rarely holds every piece of
   a multi-part answer.
8. **list_all coverage.** On dev, the top 5 holds 69.0% of a list's chunks
   (baseline 50.0%); the top 20 holds 90.5%. Long lists span more chunks than
   the prompt takes; this needs more chunks for list questions, or neighbour
   expansion around a retrieved list chunk.
9. **Refusal threshold, not wired in.** The reranker's top-1 score separates
   unanswerable from answerable questions better than cosine (AUROC 0.886 vs
   0.771 on dev; 0.935 vs 0.796 on holdout). A threshold of about -7.5 keeps
   107/113 answerable and flags 6/14 unanswerable on dev (44/49 and 3/5 on
   holdout). `src/reranker.REFUSAL_SCORE_HINT` records it; nothing uses it yet.
10. **`smoke_repo_retrieval.py` line-number check fails.** Step 4/6 ("reported
    line numbers match file contents") reports 2367 chunks verified and 2
    mismatched, so the script ends 21 passed, 1 failed. It fails the same way
    on `main` before Phase 1+2. It concerns chunk `start_line` metadata
    (chunking), not retrieval ranking. Not investigated or fixed yet.
11. **Paraphrase Hit@1 and BM25 (dev, later).** In the BM25 ablation
    (docs/DECISIONS.md), removing BM25 raised paraphrase Hit@1 by 8.6 points
    (3 questions) and MRR@5 by 7.2 (significant), while costing 6.5 points of
    Recall@20 overall. BM25 stays.
    * Follow-up, dev only: weight the fusion toward dense, or send the top
      20 of each list into the reranker instead of the fused top 20.
    * Accept only if paraphrase Hit@1 rises without losing Recall@20 on
      identifier, misspelled or multi-hop questions.
12. **First-question rerank timeout on one vCPU.** In the single-thread
    ablation run, the first question (eid-01) hit the 3 s rerank budget even
    after the startup warm-up, and fell back to the fused order. Its gold was
    at rank 1 either way, so no scores changed. Production logs show how
    often it happens.
    * Count `fallback=timeout` against all retrieval lines for
      `resource.labels.service_name="rag-qa"` (query in the Phase 2 PR).
    * If it clusters on the first question after a cold start, warm up with a
      realistic batch (20 passages) rather than one pair.

