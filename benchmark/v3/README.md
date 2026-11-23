# Retrieval benchmark v3

v3 measures **the retrieval path the app actually runs**, against **span-level
ground truth**, **deterministically**, with **paired confidence intervals**. It
replaces `evaluate_retrieval.py` (v1/v2) as the baseline for retrieval changes.
No Gemini call is made: retrieval happens before generation.

> **Status: the query set is a DRAFT under review.** It is not frozen and has
> not been scored. Scoring refuses to run until the set is frozen
> (`manifest.json`), so no retrieval result can influence the labels.

## Why v3

A diagnosis of the v2 evaluation (November 2026) found four problems that made
its "no significant difference" result uninformative:

| Problem in v2 | v3 answer |
|---|---|
| `benchmark/results.json` predates the README pin: no stored ranking carries `readme_boost`, and the shipped retriever scores 71/83, not 72/83 | The `production` arm calls `build_hybrid_retriever` exactly as `src/rag_chain.py` does, so whatever ships is measured |
| Code gold was file-level: a median of 80 chunks counted as a hit, so code Hit@5 sat near its ceiling | Span-level gold for every query: a median of 2 gold chunks per code query |
| Dense rankings changed between builds and between processes (HNSW) | The eval store uses exhaustive HNSW settings: identical production output across two independent builds for 181/181 questions |
| 83 answerable queries, so only differences of about 15 points were detectable | 162 answerable queries (draft), paired bootstrap intervals, Recall@20 reported separately from ranking quality |

## Files

| File | What it is |
|---|---|
| `queries.yaml` | The query set: v2's 93 queries (wording unchanged) plus 88 new ones |
| `REVIEW.md` | Generated review sheet: every new query and every new label, with the exact text each span covers |
| `corpus.py` | Loads the v2 corpus through the real ingestion and chunking code; verifies the repository by a content digest |
| `gold.py` | Span gold: anchors, evidence checks, parts, near-duplicates, rank metrics |
| `dataset.py` | Loads and validates `queries.yaml` (`python -m benchmark.v3.dataset`) |
| `review.py` | Regenerates `REVIEW.md` (`python -m benchmark.v3.review`) |
| `stats.py` | Paired percentile bootstrap |
| `evaluate.py` | The harness (`python -m benchmark.v3.evaluate`) |

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
  the production arm this uses the same retriever with `final_k=20`.
* **File-level Hit@5** for code queries: the v2-comparable secondary metric.

A chunk that contains at least 80% of a gold chunk's 5-word shingles also counts
as gold. That covers the .docx's duplicated block and docstrings repeated on two
methods. Symmetric containment and embedding cosine were tried and rejected,
because each admitted chunks that do not contain the answer (details in `gold.py`).

## Arms

| Arm | What it is |
|---|---|
| `production` | `build_hybrid_retriever(store).invoke(q)`: what the app returns (final_k=5, README pin on) |
| `production_no_pin` | the same fused list without the README pin |
| `dense_fallback` | the retriever's MMR vector retriever alone: what production returns if BM25 is unavailable (10 chunks) |
| `dense_plain` | plain similarity search on the same store |
| `bm25` | the retriever's BM25 index |

All arms run in one process on one store. Every aggregate is reported per
split, domain (code / docs) and query type, with paired differences against
`production`.

## Running

```bash
python -m benchmark.v3.dataset              # validate every span against the corpus
python -m benchmark.v3.review               # regenerate REVIEW.md
python -m benchmark.v3.evaluate --fidelity  # score (requires the frozen manifest)
```

`--archive PATH` points any of them at a local `psf/requests` tarball of the
pinned commit, for machines that cannot reach codeload.github.com. The archive
is accepted if its indexable files match `corpus.EXPECTED_CONTENT_DIGEST`.

`--fidelity` also builds a store with Chroma's default HNSW settings (what
production uses) and reports how many production top-5 lists differ from the
exhaustive store. Before the freeze, this was measured at 9 of 181.

## Open items

1. **Freeze** (after review): a stratified 70/30 dev/holdout split with a fixed
   seed, a SHA-256 manifest of the frozen files, and a harness that refuses to
   run if any of them change.
2. **Baseline** (after the freeze): the "before" numbers for retrieval
   Phases 1 and 2, reported separately for dev and holdout.
3. **Multi-turn query rewriting is not covered.** After the first message of a
   chat, production does not retrieve with the user's words. `src/rag_chain.py`
   first asks Gemini to rewrite the follow-up into a standalone query, and
   retrieves with that. Every v3 query is a first-turn question, so the
   rewrite step and its effect on retrieval are unmeasured. Covering it needs
   multi-turn conversations as test data and a Gemini API key to run the
   rewriter (or recorded rewrites). Not built in Phase 0.
4. **Production's default HNSW.** The eval scores an exhaustive store; production
   uses Chroma's defaults, which change 9 of 181 production top-5 lists
   relative to the exhaustive store. If those are ever judged material, the fix
   belongs in production (collection settings), not in the eval.
