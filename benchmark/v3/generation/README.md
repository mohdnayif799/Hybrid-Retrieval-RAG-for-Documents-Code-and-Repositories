# Generation eval (v4 Phase 2)

The v3 benchmark scores the chunks retrieval returns. This harness scores the
**answers**. It runs the production chain on the frozen v3 questions, then
grades every answer with an LLM judge.

> **Status:** the pilot (`results/pilot_dev.json`) chose `gemini-3.7-flash` as
> the judge. The dev baseline is `results/dev.json`. The Phase 2b comparison
> run is pending, and so are the judge-vs-human labels.

## What one question goes through

1. **The production chain, unchanged.**
   * `build_rag_chain(...).invoke(question)`, with no chat history, so this is
     a first-turn question.
   * Routing, hybrid retrieval with the cross-encoder reranker on the exact v3
     store, `_QA_TEMPLATE`, and `gemini-3.5-flash-lite` with production
     settings.
   * The only difference is the Gemini client. It is replaced by a cached,
     cost-tracked one, through `build_rag_chain(..., llm=...)`; production
     passes nothing.
2. **The judge** ([`judge_v1.md`](judge_v1.md)) returns one JSON verdict per
   field:

   | Field | Values | Judged against |
   |---|---|---|
   | `refused` | yes / no | the answer |
   | `correct` | yes / partial / no / na | the gold evidence spans, grouped into parts |
   | `grounded` (+ `unsupported_claims`) | yes / no / na | the chunks the model was actually given |
   | `length_fit` | too_short / right / too_long | what the question's wording calls for |

3. **Computed in code, not by the judge:**
   * the word count;
   * whether the exact refusal sentence appears;
   * `refusal_correct` (for unanswerable questions);
   * `wrongly_refused` (for answerable questions).

The judge prompt is pinned by SHA-256 in [`manifest.json`](manifest.json).
The runner refuses to grade with a changed rubric. A new rubric means a new
file (`judge_v2.md`), so grades from different versions never mix.

## Runs

| Command | What it does |
|---|---|
| `python -m benchmark.v3.generation.run --pilot 5` | 5 dev questions, one per type: exact_identifier, paraphrase, list_all, multi_hop, unanswerable |
| `python -m benchmark.v3.generation.run` | all 127 dev questions, plus the noise runs and the judge-check sheet |
| `python -m benchmark.v3.generation.agreement` | judge vs your hand labels on 20 questions |
| `python -m benchmark.v3.generation.run --tag NAME --noise 0` | a comparison run (for example after a prompt change): writes `results/dev_NAME.json`, never the baseline, and no judge-check sheet |
| `python -m benchmark.v3.generation.answer_features --ids A,B` | text-free features of cached answers under the baseline and current prompts (length, refusal, "says a part is missing", key facts present); cache only, $0 |
| `python -m benchmark.v3.generation.compare --tag NAME` | that run vs the baseline, question by question: Phase 2b criteria, by shape and type, flips |

**The pilot:**
* Each pilot answer is graded by flash-lite and by the stronger Flash
  candidate: `gemini-3.7-flash`, or `gemini-3.5-flash` if 3.7 is not available
  on the key.
* It prints the measured tokens and the cost per call, then projects the total
  eval spend of the whole v4 plan for each judge.
* **The judge rule:** the stronger judge is used if its projection is under
  $1.50; otherwise flash-lite.

**The full run:**
* Uses the judge the pilot chose.
* Re-judges 20 answers and regenerates 20 answers. Agreement across those
  re-runs is the noise floor that later phases' differences must clear.
* Writes the judge-check files:
  * `judge_check_dev.md` to read;
  * `judge_check_dev.yaml` to fill in.
  Both are local and gitignored, because they hold text.

**Holdout** is refused, except with `--split holdout --final-look`, used once in
Phase 6.

## Cost controls

* **Cache.** Every model call is stored on disk under
  `benchmark/.cache/generation/` (gitignored). A rerun makes zero calls and
  costs nothing. An interrupted run resumes from where it stopped.
* **Caps.** A run stops before it would spend more than `--max-usd` (default
  $0.50). The whole plan stops at `--max-total-usd` (default $2.00), counting
  every call already in the cache.
* **Throttle and retries.** `--rpm` (default 10 requests per minute).
  Rate-limit and server errors are retried with backoff.
* **Prices** are list prices from `llm.PRICES`, in USD per 1M tokens, and can
  be overridden with `--price MODEL=IN,OUT`:

  | Model | Input | Output |
  |---|---|---|
  | `gemini-3.5-flash-lite` | 0.30 | 2.50 |
  | `gemini-3.7-flash` (introductory, through 2026-12-31) | 0.75 | 3.75 |
  | `gemini-3.5-flash` | 1.50 | 9.00 |

  Thinking tokens are billed as output.

## What is written, and what is not

* **Written** to `results/` (committed): verdicts, word counts, token counts,
  timings, cost, and the hashes of the frozen question set, the judge prompt
  and the answer prompts.
* **Never written there:** question, answer or chunk text.
* Prompts and answers live only in the local cache.

## Windows

* The frozen files, the judge prompt and the corpus documents are verified by
  SHA-256.
* If Git converted their line endings to CRLF on checkout, the hashes read
  CRLF as LF.
* If the `.txt` corpus document was converted, an LF copy is used, and it
  matches the pinned hash exactly.
* `.gitattributes` keeps `benchmark/` byte-exact on new checkouts.
