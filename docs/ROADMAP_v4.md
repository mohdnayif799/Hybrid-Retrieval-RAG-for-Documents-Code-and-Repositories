# Roadmap v4: answer quality after retrieval

Status: **plan only (Phase 0)**. Nothing here is built yet. Each phase gets its
own branch and PR, cut from the latest `main`. Each stops for approval, then
merges with a merge commit once CI is green, after which the deploy and the
live `/_stcore/health` are checked.

v3 (PRs #5 and #6) made retrieval measurable and better. v4 looks at what
happens after retrieval:

* how sources are shown;
* whether answers are correct, grounded and the right length;
* how badly typed and list-style questions are handled.

## Ground rules (apply to every phase)

| Rule | How it is enforced |
|---|---|
| Decisions on **dev** only | Every new harness defaults to `--split dev`. Holdout needs `--split holdout --final-look`, used once, in Phase 6. |
| Retrieval holdout: second and final look in Phase 6 | PR #6 scored it once. Phase 6 records this as the second look in `benchmark/v3/README.md` and DECISIONS. |
| No Gemini key here | Production code takes an injectable LLM callable. Tests use a fake LLM. Real runs are laptop commands (PowerShell, key from the repo's `.env` via `load_dotenv`), and I wait for the pasted output. |
| Every eval LLM call cached | On-disk cache keyed by `sha256(model, purpose, prompt, config)`, so a rerun makes zero calls. Every run prints a cost projection first, and a hard cap aborts it. |
| Never log question, answer or chunk text | Log lines carry only counts, labels, timings and scores. Committed eval results hold verdicts, word counts and token counts, not answer text. |
| ruff + full suite pass in every phase | As today. CI runs both. |
| Each decision with its numbers | Appended to `docs/DECISIONS.md` in the phase that makes it. |

## What the code does today (read for this plan)

* **Routing** (`src/rag_chain.py:250`): if the chat has no documents, or
  `_is_casual_message` fullmatches a small regex list (`rag_chain.py:21-64`),
  the message goes to the conversational prompt without retrieval. Everything
  else goes to RAG.
* **Follow-up rewrite** (`rag_chain.py:256-267`): only when there is chat
  history. One Gemini call rewrites the message into a standalone query with
  `_CONTEXTUALIZE_TEMPLATE`, falling back to the raw question if the output is
  over 500 characters or multi-line. **First-turn questions are never
  rewritten**, so typos reach retrieval untouched.
* **Answer prompt** (`_QA_TEMPLATE`, `rag_chain.py:172-188`):
  * "use ONLY the context", plus an exact refusal sentence and a casual-message
    escape hatch;
  * the context is the 5 chunk texts joined by blank lines, with no file names
    (`rag_chain.py:270`);
  * history is included for reference resolution, with assistant turns
    truncated to 800 characters (`rag_chain.py:196`);
  * there is no length guidance at all.
* **Gemini calls per question:** 1 answer call, plus 1 rewrite call on
  follow-ups, plus 1 title call on a chat's first message (`app.py:203`,
  `app.py:814`). The default model is `gemini-3.5-flash-lite`
  (`app.py:174`).
* **Response cache** (`app.py:827-850`): a per-chat dict keyed by
  `question.strip().lower()`, storing the answer and source docs. It is
  cleared only when the store is evicted (`app.py:510`). See *Findings
  outside the phases*.
* **Sources display:**
  * `_render_sources` (`app.py:275`) draws a "Sources:" citation list plus a
    "Source Chunks Used" expander.
  * The replay loop (`app.py:784-795`) calls it for **every** assistant
    message, so a long chat repeats a full sources block under every answer.
  * Sources are stored on each message (`app.py:861-866`), so each answer
    already owns its sources. Only the display changes in Phase 1.
* **Per-chat stores:**
  * Each chat has its own Chroma dir (`store_paths.py`), registered for
    idle/LRU eviction (`store_registry.py`, `vector_store.py:116`).
  * The BM25 index is cached per `chroma_dir` (`hybrid_retrieval.py`) and
    forgotten on eviction.
  * Every chunk carries `source`, `start_index` (uploads too: 111/111 in the
    v3 corpus) and, for repositories, `start_line`. Adjacent chunks and whole
    documents can therefore be rebuilt from the store without new metadata.

## Cost and latency model

**Prices.** `gemini-3.5-flash-lite` at **$0.30 per 1M input tokens and $2.50 per
1M output tokens**. These are third-party listings of Google's API rates
([Artificial Analysis][aa], [DeepMind model page][dm]). The context window is
1,048,576 tokens ([Google Cloud docs][gc]). Prices are confirmed in Phase 2
from the token counts Gemini returns (`usage_metadata`). Gemini 3.x bills
thinking tokens as output. If flash-lite thinks by default, real output cost
could be 2-4x the figures below. Phase 2's pilot run measures it before any
full run.

**Token estimate.** 4 characters ≈ 1 token.

| Item | Tokens |
|---|---|
| Document chunk (≤1,000 chars) | ≈ 250 |
| Code chunk (≤800 chars) | ≈ 200 |
| Answer prompt today (template + 5 chunks + question) | ≈ 1.5k |
| Answer output | ≈ 150-300 |

**Today, per question (estimates; Phase 2 measures them):**

| | Gemini calls | Cost | Latency on rag-qa (1 vCPU) |
|---|---|---|---|
| First question of a chat | title + answer | ≈ $0.0011 | retrieval ≈ 1.1-1.4 s (rerank p50 1.04 s), plus Gemini answer ≈ 1-3 s (unmeasured) |
| Later first-turn question | answer | ≈ $0.0010 | same |
| Follow-up | rewrite + answer (prompt carries history) | ≈ $0.002 | + one more Gemini round trip (≈ 0.5-1.5 s) |

**Eval spend projection (all phases):**

| Phase | What is called | Calls | Estimate |
|---|---|---|---|
| 2 | Dev baseline: 127 answers + 127 judgements, plus 20 re-judged and 20 re-generated as noise checks | ≈ 300 | ≈ $0.32 |
| 3 | LLM-rewrite comparison: 127 rewrites, retrieval only | 127 | ≈ $0.02 |
| 4 | Generation eval on dev list, multi-part and small-document questions only (≈ 50) | ≈ 100 | ≈ $0.10 |
| 5 | Dev re-run with the length prompt (unchanged answers re-use cached judgements) | ≈ 254 | ≈ $0.30 |
| 6 | Holdout, baseline arm + final arm: 54 × 2 answers and judgements | ≈ 216 | ≈ $0.25 |
| **Total** | | | **≈ $1.00** (cap: abort any single run projected over $0.50) |

That is about a dollar if thinking tokens are small, which is above "in the
cents" taken strictly. You accepted about $1 (see *Decisions taken*).

If your key is on the free tier, the cost is $0 but rate limits apply. The
runner throttles to a configurable requests-per-minute, and resumes from the
cache after a 429, so an interrupted run costs nothing to restart.

---

## Phase 1: Sources toggle (UI only)

**Goal.** The latest answer shows its sources as today. Each earlier answer
keeps its own sources in a collapsed "Sources" expander underneath it.

**Files.** `app.py` (the replay loop and `_render_sources`),
`tests/test_app_ui.py`, a browser-check script under `scripts/` (not part of
the suite), `docs/DECISIONS.md`.

**Design.**
* `_render_sources(docs, collapsed: bool)`. Expanded mode is exactly today's
  markup.
* Collapsed mode draws one `st.expander(f"Sources ({n})", expanded=False)` with
  the citation list and the chunk details. Streamlit does not allow an
  expander inside an expander, so in collapsed mode the chunk details are
  drawn inline inside the single expander instead of in a nested "Source
  Chunks Used" expander.
* **Latest** = the last assistant message in the chat. If it has no sources
  (casual reply, refusal from an error), nothing is expanded.
* The replay loop runs before the live answer. If a new question arrived in
  this run, every replayed answer is "earlier" and drawn collapsed. The
  previous answer therefore collapses as soon as the next question is sent,
  not one rerun later.
* Cached answers behave like live ones. Error messages carry no sources.
* No retrieval, prompt or cache changes.

**Tests.**
* AppTest with the existing fake Gemini client (`tests/test_app_ui.py:39`) and
  a monkeypatched retriever returning fixed chunks. Two questions are asked,
  then:
  * the first answer has exactly one collapsed "Sources" expander;
  * the second shows the full block;
  * each expander holds its own answer's citations;
  * a third question collapses the second.
* Browser check: Playwright (pre-installed here) drives `streamlit run` on a
  launcher that patches the Gemini client. It uploads a small `.txt`, asks two
  questions, clicks the earlier "Sources" expander, and takes a screenshot.

**Risks.**
* AppTest's expander API does not expose the expanded state on every
  Streamlit version. If so, the test asserts on labels and contents, and the
  browser check covers the expanded state.
* Long chats render many expanders. They are collapsed, so the cost is markup
  only.

**Latency and cost per question.** None (display only).

---

## Phase 2: Generation eval harness

**Goal.** Measure answers, not just chunks, on the frozen v3 questions, cheaply
and repeatably.

**Files.**
* New `benchmark/v3/generation/`:
  * `run.py`: CLI; pilot, projection, cap, throttle;
  * `llm.py`: cached Gemini wrapper, token and cost accounting;
  * `judge.py`;
  * `judge_v1.md`: prompt and rubric;
  * `manifest.json`: SHA-256 of `judge_v1.md`, `run.py`'s prompt-building
    code and the model names, checked like the v3 freeze;
  * `judge_check.py`: builds the 20-question hand-label sheet;
  * `agreement.py`;
  * `report.py`;
  * `README.md`.
* `src/rag_chain.py`: `build_rag_chain(..., llm=None)`, an optional injected
  callable. When omitted it is today's `google_genai` client, so production
  behaviour does not change.
* `tests/test_generation_eval.py`: fake LLM, no network.
* `.gitignore`: the local cache dir.
* `docs/DECISIONS.md`.

**Design.**
1. **Chain.**
   * Questions: the frozen v3 dev questions (127: 113 answerable, 14
     unanswerable).
   * Store: the same exhaustive v3 store the retrieval harness builds, so
     retrieval is deterministic.
   * Each question runs through `build_rag_chain(...).invoke({"input": q,
     "chat_history": []})`: production routing, retrieval with the reranker,
     `_QA_TEMPLATE`, and `gemini-3.5-flash-lite` with production settings.
   * The chunks the model actually received are captured from
     `result["context"]`.
2. **Judge.**
   * `gemini-3.5-flash-lite`, temperature 0, JSON output against a schema,
     thinking at its lowest setting.
   * One call per question that returns one verdict per field:
     * `correct`: yes / partial / no / n/a. Is the answer supported by the gold
       **evidence spans**? The judge sees the question, the gold evidence text
       and the answer. For multi-part gold, partial means some parts are
       missing.
     * `grounded`: yes / no. Is every claim in the answer supported by the
       **retrieved chunks**? The judge sees the retrieved chunks, not the gold.
       It also lists unsupported claims by index only.
     * `refusal_correct`: yes / no / n/a. Unanswerable: did it refuse?
       Answerable: did it wrongly refuse? A deterministic check for the exact
       refusal sentence is recorded beside the judge's verdict.
     * `length_fit`: too_short / right / too_long, against a written rubric
       keyed on question wording: a fact gets 1-3 sentences, a list gets every
       item with little prose, an explanation gets one or two short
       paragraphs, a summary gets structured key points.
     * `word_count` is computed in code, not by the judge.
3. **Versioning.**
   * `judge_v1.md` holds the prompt and rubric. Its hash and the
     answer-prompt hash are written into every result.
   * The runner refuses to mix results across hashes. Changing the rubric
     means `judge_v2.md` and a re-judge (cheap: answers are cached).
4. **Caching and cost.**
   * Each call is cached on disk (JSONL, gitignored) under
     `sha256(model, purpose, prompt, generation-config)`.
   * The first command is a 5-question **pilot**. It prints measured tokens
     (input, output and thinking), seconds per call, and the projected cost
     of the full run.
   * The full run aborts if its projection exceeds the cap.
5. **Noise.**
   * 20 answers are re-judged once, giving judge self-consistency.
   * 20 answers are re-generated once, giving the generator flip rate.
   * These two numbers set the smallest difference later phases can claim.
6. **Judge check.** The run writes `judge_check_dev.yaml` locally: 20 dev
   questions stratified by question type, each with question, gold evidence,
   retrieved chunks and answer, and blank fields for your labels. You label
   them. `agreement.py` then prints raw agreement and Cohen's kappa per field,
   numbers only, which you paste back. Proposed bar: ≥ 80% raw agreement on
   `correct` and `grounded`. Below that, the rubric is revised (v2) before any
   phase relies on the judge.
7. **Output.**
   * Committed `results_gen_dev.json`: per question, verdicts, word count,
     tokens, seconds, refusal flag, plus hashes. No question, answer or chunk
     text.
   * `RESULTS_generation.md`: per question type rates with bootstrap CIs.

**Laptop commands (PowerShell, from the repo root, key in `.env`)**, given
exactly in the Phase 2 PR:
```
git fetch origin
git checkout claude/clever-bardeen-mwqega
git pull
python -m benchmark.v3.generation.run --split dev --pilot 5
python -m benchmark.v3.generation.run --split dev
```
* The first run downloads the pinned `psf/requests` archive and builds the
  eval store, taking a few minutes.
* Each command ends with a summary of about 30 lines to paste back.
* Optionally, push `results_gen_dev.json` (it contains no text) so I can do
  per-question analysis here.

**Risks.**
* **Self-judging:** flash-lite judging flash-lite can be lenient. Mitigations:
  * the strict rubric with gold spans;
  * the human check;
  * the option to judge with `gemini-3.5-flash` instead (≈ 3-4x the judge
    cost, still cents) if agreement is low.
* **Small cells:** misspelled 7, list_all 7, repo_identity 4, unanswerable 14
  on dev. Per-type CIs will be wide; decisions lean on all-dev numbers and
  paired flips.
* **Unmeasured follow-ups:** first-turn only, as in v3 (open item 3).
* **Thinking tokens:** unknown in advance. The pilot measures them.

**Latency and cost per question.** None in production; it is eval only.

---

## Phase 2b: Over-refusal fix (added 2026-11-28, before Phase 3)

**Why.** The Phase 2 dev baseline is grounded (99%) and refuses every
unanswerable question (14/14). But it wrongly refused **26 of 113 answerable
questions**. The model is over-cautious, not hallucinating.

**Diagnosis of the 16 the baseline flagged as "evidence in context".** Each
context was rebuilt from the same deterministic store and checked
structurally, with no text printed:
* the question's shape;
* which gold chunks, including near-duplicate copies, were among the 5
  chunks the model saw;
* how many answer parts they covered.

| cause | n | questions | what the model had |
|---|---|---|---|
| **A. Bare term**: 1-3 words, no "?", an identifier or keyword | 8 | eid-01, eid-02, eid-04, eid-10, isw-13, lit-01, lit-05, msp-03 | the gold chunk at rank 1-4 and the only part; all code |
| **B. Multi-part, some parts present** | 2 | crs-05, mh-02 | 1 of 2 parts in context (gold at rank 1) |
| **C. Full question, everything present** | 2 | mh-06 (2/2 parts, ranks 3 and 5), pp-03 (rank 1, docs) | the whole answer |
| **D. Retrieval miss**, miscounted | 4 | isw-01, isw-10 (bare identifiers), msp-06 (misspelled), par-05 | no gold chunk; the evidence string occurred in other chunks |

So 12 are answer-step failures: A, B and C. The 4 in D are retrieval misses
for Phase 3 (typos) and later. The harness now records `gold_in_context` and
`parts_in_context` instead of the loose evidence-string check. The other 10
of the 26 failed even the loose check; their split under the new definition
comes with the regenerated dev.json.

* **Cause A** is the dominant one. The prompt is written for questions; given
  a bare name, the model takes the refusal path, even with the name's
  definition at rank 1. 29 of the 127 dev questions have this shape.
* **Cause B:** the prompt's only options are a full answer or the refusal
  sentence, so a half-answerable question gets refused whole.
* **Cause C** suggests the rule "if the context does not contain the answer"
  is read too strictly when the answer needs a small inference across chunks.

**The proposed change** is the smallest that addresses A, B and C. Only the
refusal rule of `_QA_TEMPLATE` changes:

```diff
 Use ONLY the context below to answer the question.
-If the context does not contain the answer, say exactly:
+If the question is only a term, name or identifier rather than a full question,
+treat it as "What is <term> and what does it do?" and answer that from the context.
+If the context answers only part of the question, answer that part and say plainly
+which part the documents do not cover.
+Only if the context contains nothing that answers any part of the question, say exactly:
 "I don't have enough information in the provided documents to answer this."
-Do not fabricate any information.
+Do not fabricate any information: every statement must be supported by the context.
```

The rest of the prompt is unchanged: the exact refusal sentence, the casual
escape and the history block. No retrieval change, no extra model call.

**Measurement** (dev only; laptop command):
* The generation eval adds `--tag` (it writes `dev_<tag>.json`), plus a
  comparison script with paired bootstrap differences against
  `results/dev.json`, per question type and per shape, and the questions that
  flipped.
* The new prompt changes every answer prompt, so all 127 answers and grades
  are new: ≈ 260 calls, ≈ $0.35, plus ≈ $0.08 for the noise re-runs.
* Spent so far: $0.37 of the ≈ $1.17 plan. Phase 2b's ≈ $0.43 is extra, which
  takes the projected total to ≈ $1.6. Phase 5 can reuse this run's design to
  stay cheap; see Risks.

**Success criteria (dev, all four required):**
1. **`correct` goes up** (yes rate, and the yes=1 / partial=0.5 score),
   paired against the baseline.
2. **Wrong refusals with gold in context fall** from 12. Target ≤ 3; cause A
   should go to ~0.
3. **Unanswerable refusals stay 14/14.**
4. **Grounded stays ≥ 97%.**

Also reported, not gating: length_fit (bare terms will get longer answers),
flips in each direction, and the noise floor from the Phase 2 noise runs.

**Risks.**
* **Unanswerable questions may get partial or tangential answers.** "Answer
  the part you can" could make the model offer related facts where it should
  refuse. That is criterion 3.
  * The judge counts a reply whose substance is "the documents don't cover
    this" as a refusal.
  * A single-part unanswerable question has no answerable part.
* **Bare terms may get over-long answers.** A bare term that names something
  minor could draw a long explanation. length_fit tracks this; Phase 5 owns
  answer length.
* **Budget.** The extra run pushes the projected total over $1.50. Phase 5
  (answer length) re-generates every answer again. To stay near the budget,
  Phase 5 can be measured on the questions whose intent changes the prompt
  line (fact and list, ≈ 80), or the two prompt changes can be measured
  together. That is your call when Phase 5 comes.

**Latency and cost per question in production.** No extra calls. Answers to
bare terms and to partial questions get longer (output tokens), so
≈ +$0.0002-0.0005 on those questions.

---

## Phase 3: Query understanding (typos and intent)

**Goal.**
* Fix misspelled words before retrieval, for free.
* Label each question's intent (fact / list / explain / summary) for Phases 4
  and 5.

**Files.**
* New `src/query_understanding.py`: the corrector and the intent rules.
* `src/hybrid_retrieval.py`:
  * the vocabulary is built and cached with the BM25 index, per `chroma_dir`,
    and forgotten with it;
  * `HybridRetriever(correct_typos=True)` corrects the retrieval query only;
  * the log line adds `typo_fixes=N intent=list`, counts and labels only.
* `src/rag_chain.py`: intent is passed through.
* `benchmark/v3/evaluate.py`: new arms; fingerprint handling, see Risks.
* New `benchmark/v3/typo_variants.py`.
* New `benchmark/v3/intent_labels.yaml`: dev only.
* `tests/test_query_understanding.py`.
* `benchmark/v3/generation/rewrite_compare.py`.
* `docs/DECISIONS.md`.

**Typo correction design.**
1. **Vocabulary.** Token frequencies from the chat's own BM25 tokenization.
   For the v3 corpus that is 8,021 terms, 5,583 of them alphabetic with 4 or
   more letters, built in 0.12 s at index time.
2. **Which query words are candidates.** All of these must hold:
   * alphabetic and 4+ letters;
   * not in the chat vocabulary;
   * not in a bundled list of common English words (≈ 30k, MIT/public-domain
     source), so a correctly spelled word the corpus doesn't use, such as
     "explain", is not "corrected" into a corpus word;
   * not an identifier: no `_`, camel case, digits, `.`, `()` or backticks;
   * not a number.
   Words that exist anywhere in the vocabulary are never touched.
3. **Candidates.**
   * Damerau-Levenshtein distance ≤ 1 for words of 4-7 letters, ≤ 2 for 8+.
   * Ranked by distance, then corpus frequency.
   * Corrected only if the best candidate is unique at that distance, or at
     least 3x more frequent than the runner-up. Otherwise the word is left
     alone.
4. **Speed.**
   * Brute force in pure Python measured **30-40 ms per misspelled word**
     here. That is too slow for the 20 ms target.
   * So a SymSpell-style index is used: deletes at distance 1 of each vocab
     word (≈ 50-65k keys for this corpus), plus query-side deletes at distance ≤ 2,
     verified with Damerau-Levenshtein.
   * Expected cost: under 1 ms per word, and a few MB per store.
   * The index is built lazily on the first query that has a candidate word,
     and cached and evicted with the BM25 index.
   * Budget: < 20 ms per query end to end, asserted in a test on a
     1,000-chunk store.
5. **Scope of the change.**
   * Only the retrieval query (dense, BM25 and reranker input) is corrected.
     The answer prompt keeps the user's original words.
   * Follow-ups are corrected after the existing LLM rewrite.

**Intent design.**
* Ordered regex rules on the question wording:
  * **summary:** summarize / summary / overview / "what is this (doc|repo|project) about" / key or main points / tl;dr;
  * **list:** list / "what are (all) the" / name all / enumerate / "which … are" / types or kinds of / "how many and which";
  * **explain:** how does / why / explain / describe / "what happens when" / difference between / compare;
  * **fact:** everything else.
* Runs after the casual router, so casual messages never get an intent.
* Accuracy is checked against hand labels for the 127 dev questions
  (`intent_labels.yaml`). I draft the labels; you review them in a short sheet.

**Measurement (no key).**
* The retrieval eval on dev, production vs production + typo correction,
  paired.
* **The misspelled set is tiny (7 dev questions).** So also a deterministic
  **synthetic typo set**: every answerable dev question gets one seeded
  misspelling of one content word (swap, drop, double or substitute). That
  gives ≈ 100 variants, measuring how much Hit@5 typos cost and how much
  correction recovers.
* Guard: the number of non-misspelled dev questions the corrector changes at
  all (target: 0). Any changed one must keep its Hit@5.
* Typo twin table, as in v3.

**LLM rewrite comparison (laptop command).**
* One flash-lite call per first-turn question: fix typos and make the query
  standalone. Cached.
* The same retrieval metrics on the same dev sets.
* The recommendation weighs quality against latency (+0.5-1.5 s, one more
  call) and cost (≈ $0.0001 per question, plus one more request against
  free-tier quotas). It ships only if it beats the deterministic corrector by
  more than the noise, on the misspelled + synthetic sets, without hurting
  the rest.

**Risks.**
* **Over-correction.** Domain words absent from both lists get "fixed"; this
  is guarded by the change count on clean questions and the 3x-frequency
  margin.
* **Non-English corpora.** The English list does not help them, but the
  corpus vocabulary still protects their real words.
* **Memory.** Index per store × up to 40 stores. Measured in-phase; the lazy
  build means stores that never see a typo pay nothing.
* **Holdout fingerprint.**
  * The v3 freshness test requires `results.json` to match the retrieval
    code. Changing retrieval would force a rerun that includes holdout, which
    would be a holdout look before Phase 6.
  * Fix: from Phase 3, the freshness test checks a **dev-only**
    `results_dev.json`. PR #6's `results.json` is renamed
    `results_phase12.json` as a historical record.
  * Holdout is scored again only in Phase 6.
* **Intent rules are English-only.** Misses fall back to "fact", which is
  today's behaviour.

**Latency and cost per question.**

| | Latency | Cost |
|---|---|---|
| Typo correction | < 20 ms (target; measured) | $0 |
| Intent | < 1 ms | $0 |
| LLM rewrite (comparison only) | +0.5-1.5 s | ≈ +$0.0001 |

---

## Phase 4: List and summary questions

**Goal.** Long lists and summaries arrive whole instead of cut at 5 chunks. On
dev, list_all coverage is 69.0% @5 and 90.5% @20, and multi-part all-parts@5 is
20%.

**Files.**
* `src/rag_chain.py`: small-document route, intent-dependent context.
* `src/hybrid_retrieval.py`: `final_k` by intent, neighbour expansion.
* `src/vector_store.py`: rebuild documents and neighbours from `start_index`.
* `app.py`: sources label for whole-document answers.
* `benchmark/v3/evaluate.py`: arms.
* The generation runner: arm flags.
* Tests.
* `docs/DECISIONS.md`.

**Small-document mode.**
* If the chat's documents total ≤ **8,000 tokens** (≈ 32k characters,
  estimated as characters / 4 from the store), retrieval is skipped. The model
  gets each whole document in order, rebuilt from chunk `start_index` with
  overlaps removed and labelled by file name.
* **Why 8k:**
  * The input becomes at most ≈ 5x today's (≈ $0.0024 per question, versus
    ≈ $0.0005 for 5 chunks).
  * It saves ≈ 1.1-1.4 s of reranking, against ≈ 0.3-0.8 s more prefill, so
    the net effect is latency-neutral or faster.
  * It is far below flash-lite's 1M context, so there are no long-context
    effects.
  * It covers documents of up to roughly 12-15 pages.
* **Coverage in the v3 corpus:**

  | Document | Tokens | Small-document mode? |
  |---|---|---|
  | PDF | ≈ 650 | yes |
  | `.txt` | ≈ 1.7k | yes |
  | `.docx` | ≈ 19.6k | no (retrieval) |
  | `psf/requests` | ≈ 180k | no (retrieval) |

* **The eval needs a per-document view.** The v3 store mixes all sources
  (1,222 chunks), so it never triggers this mode. The arm therefore asks each
  docs question against a store holding only its own document, which is how a
  user who uploads one file sees the app.
* 24k (which would include the `.docx`) is reported as a sensitivity check
  only if you want it: ≈ +$0.25 of eval.
* **Sources.** With no retrieval there are no ranked chunks, so the sources
  block names the document(s) and pages used ("whole document"). The
  alternative is to keep retrieval running for the citation display only
  (+1.1-1.4 s). Decided: no retrieval, with page or section markers the model cites (see *Decisions taken*, item 4).

**List/summary on larger stores.**
* For list or summary intent, return the reranker's top **10** instead of 5.
  The pool's top 20 is already scored, so this adds no rerank time.
* Then add the neighbouring chunks (±1, up to ±2 while a list continues
  across the boundary) of the top 3. The neighbours must be from the same
  file, and the same page for PDFs. Overlaps are merged.
* Cap: 12 chunks, ≈ 3k tokens.
* k ∈ {5, 8, 10, 12} × expansion on/off is swept on dev. The choice is made
  by coverage per context token, not raw coverage, so "more chunks" has to pay
  for itself.

**Measurement.**
* **Retrieval eval (dev, no key):** list_all chunk coverage @k and
  all-parts@k on multi-part questions, plus Hit@1/5 for every other type, to
  confirm the intent gating leaves them alone.
* **Generation eval (laptop):** dev list, multi-part and small-document
  questions (≈ 50): correct, grounded, refusal_correct.

**Risks.**
* **Intent errors.** A fact question tagged "list" gets more context. That
  costs tokens and may lengthen the answer; it is tracked by `length_fit` in
  Phase 5.
* **History truncation.** The 800-character truncation of assistant turns in
  history (`rag_chain.py:196`) will cut long list answers in follow-ups.
  Noted, not changed here.
* **Grounding in whole-document mode** is judged against the whole document
  the model saw. The judge input grows (≤ 8k tokens), which is still cents.

**Latency and cost per question.**

| | Latency | Cost |
|---|---|---|
| Small-document mode | −1.1 s to +0.5 s net | +≤ $0.002 |
| List/summary k=10 + neighbours | +5-20 ms (Chroma lookup), +0.2-0.5 s prefill | +≈ $0.0006 |
| Everything else | unchanged | unchanged |

---

## Phase 5: Answer length by intent

**Goal.** Answer length follows the intent: short facts, complete lists,
moderate explanations, structured summaries.

**Files.** `src/rag_chain.py` (`_QA_TEMPLATE` gains one intent-specific length
line; the exact refusal sentence is unchanged), tests, generation-eval re-run,
`docs/DECISIONS.md`.

**Design.** One line per intent, with targets calibrated from the Phase 2
baseline's `length_fit` and word counts:
* **fact:** 1-3 sentences;
* **list:** every item, one line each, no padding;
* **explain:** one or two short paragraphs;
* **summary:** key points as bullets.

**Measurement (laptop).** The full dev generation eval, paired against the
Phase 2 baseline. Proposed acceptance (please confirm):
* `length_fit` = right improves;
* `correct` and `grounded` do not drop: the point estimate is not below
  baseline, and yes→no flips are no more than no→yes flips plus the measured
  noise rate.

**Risks.**
* Shorter fact answers may drop qualifiers. That is caught by `correct`
  partial/no.
* The prompt change shifts every answer, so a full re-run is needed (≈ $0.30).

**Latency and cost per question.**
* Fewer output tokens for facts: faster and cheaper (output is 8x the input
  price).
* More output for lists. Net effect measured.

---

## Phase 6: Final check

* **Retrieval holdout, scored once: its second and final look** (the first was
  PR #6). It is reported against `baseline_v3` and against PR #6's production
  (`results_phase12.json`).
* **Generation eval on holdout (54 questions), once, laptop command.** Two
  arms in the one run:
  * the Phase 2 baseline configuration, with all v4 flags off;
  * the final configuration.
  Every v4 feature is behind a flag in one `PipelineConfig`, so the baseline
  stays reproducible.
* **Report:** dev vs holdout, each against the baseline, with paired bootstrap
  CIs, per question type.
* **Docs:** update `README.md`, the `benchmark/v3/README.md` open items, and
  `DECISIONS.md` (including the holdout-look count). Then stop.

---

## Findings outside the phases

1. **The response cache could serve stale answers.** It was keyed only by the
   lowercased question, was never cleared when documents were added, and
   ignored chat history. **Fixed in its own PR before Phase 1**: the key is
   (chat, document-set digest, question), and follow-ups that depend on
   earlier turns bypass the cache. See `src/response_cache.py` and
   DECISIONS.md.
2. **The answer context has no file labels** (`rag_chain.py:270`). The model
   can't say which file a fact came from, and the UI shows citations instead.
   Labelling chunks could help multi-file answers. Not planned; noted.
3. **The `smoke_repo_retrieval.py` line-number check** (v3 open item 10)
   remains out of scope.

## Decisions taken (2026-11-24)

These override the phase descriptions above where they differ.

1. **Budget.** About $1 total for all eval runs is accepted. The Phase 2 noise
   re-runs stay.
2. **Judge model and bar.**
   * The judge is `gemini-3.5-flash` if the 5-question pilot projects the
     total eval spend under $1.50; otherwise it is flash-lite.
   * The judge passes if it agrees with your hand labels at least 80% of the
     time on `correct` and on `grounded`.
   * The agreement report also gives the direction of each disagreement:
     whether the judge was more lenient or stricter than you.
3. **Results files come back by git.** For every laptop step, the PR gives
   exact PowerShell commands, from the repo folder, to:
   * fetch, check out and pull the branch;
   * run the eval;
   * `git add` only the results file (verdicts and counts, no text), commit,
     and push to the same branch.
   I stay stopped until you say to pull and continue.
4. **Small-document mode uses no retrieval.**
   * The full-document context carries page (PDF, PPTX) or section markers.
   * The model is told to cite those markers.
   * The sources block shows the document name and the pages or sections the
     answer cited.
5. **Response-cache fix:** its own PR before Phase 1, keyed by chat, document
   set and question, with follow-ups that depend on history skipping the
   cache. Tests cover a document added after a cached answer, and the same
   question in two chats.
6. **Synthetic typo set (Phase 3):** generated only from dev questions, with a
   fixed seed, and **frozen (hashed) before anything is measured on it**.
7. **Freshness test:** from Phase 3 it points at dev-only results until
   Phase 6.

[aa]: https://artificialanalysis.ai/models/gemini-3-5-flash-lite
[dm]: https://deepmind.google/models/gemini/flash-lite/
[gc]: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-flash-lite
