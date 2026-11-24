# Retrieval decisions

Why the retrieval pipeline looks the way it does, with the numbers behind each
choice. All measurements are from the v3 benchmark (`benchmark/v3/`): 162
answerable and 19 unanswerable queries over a 1,222-chunk corpus. The set was
frozen into a 70/30 dev/holdout split before anything was scored.

Decisions were made on **dev** (113 answerable queries). **Holdout** (49) was
scored once, after the configuration was fixed, to confirm. Differences are
paired, with 95% bootstrap intervals; `*` means the interval excludes 0. Full
tables are in [`benchmark/v3/RESULTS.md`](../benchmark/v3/RESULTS.md).

## Phase 1+2 (2026-11): candidate pool + cross-encoder reranker

### The result

| | dev baseline | dev now | dev difference | holdout baseline | holdout now | holdout difference |
|---|---|---|---|---|---|---|
| Hit@1 | 39.8 | 54.9 | **+15.0** [+6.2, +23.9] * | 44.9 | 55.1 | +10.2 [−6.1, +26.5] |
| Hit@5 | 73.5 | 81.4 | **+8.0** [+1.8, +15.0] * | 73.5 | 89.8 | **+16.3** [+6.1, +28.6] * |
| MRR@5 | 52.6 | 64.4 | **+11.8** [+5.3, +18.1] * | 56.0 | 69.9 | **+13.9** [+3.2, +24.8] * |
| Recall@20 | 84.2 | 89.5 | **+5.3** [+1.3, +9.7] * | 81.3 | 89.5 | **+8.2** [+2.0, +16.3] * |
| all-parts@5 | 69.9 | 76.1 | +6.2 [0.0, +12.4] | 65.3 | 83.7 | **+18.4** [+8.2, +30.6] * |

Code gains most: dev code Hit@1 23.1 → 41.5 (+18.5 [+9.2, +29.2] *), holdout
code Hit@5 64.3 → 82.1 (+17.9 [+3.6, +32.1] *). Document queries were already
strong (dev Hit@5 87.5) and improve without reaching significance on dev.

### 1. MMR removed from the dense side

The old dense retriever was LangChain's Chroma MMR (k=10, fetch_k=30,
lambda=0.6). LangChain returns the MMR *selection* in plain similarity order
(`[r for i, r in enumerate(candidates) if i in mmr_selected]`), so MMR never
reordered anything. It only dropped candidates. On dev, the MMR output and
plain similarity had identical Hit@1 (46.9 each), and MMR removed a gold chunk
from the top 5 on 4 queries while adding one on 3. With a reranker choosing
the final order, diversity filtering before the pool only costs recall.
**Now:** plain similarity, top 50.

### 2. A recall-oriented pool: plain50 + BM25-50, min-max fusion

With a reranker, fusion no longer has to put the best chunk in the top 5, only
in the reranker's window. The pool was therefore chosen on dev Recall@20/30:

| pool (dev) | Recall@20 | Recall@30 |
|---|---|---|
| old: MMR 10 + BM25 10, RRF k=60 (at most 20 candidates) | 84.2 | 84.2 |
| plain50 + BM25-50, RRF k=60 | 86.4 | **91.3** |
| **plain50 + BM25-50, min-max α=0.5** | **89.5** | 90.4 |

Min-max won at 20, the reranker's window, and was within one query of the
best at 30. After reranking, the two pools gave identical Hit@1 and Hit@5;
min-max kept more of each list (list coverage@5 69.0 vs 64.3). Depths 20, 30
and 50 per retriever were compared; 50 was best or tied.

### 3. BM25 tokenizer keeps whole identifiers

Identifiers are indexed whole as well as split, and stopwords are removed only
outside identifiers: `TooManyRedirects` → `toomanyredirects, too, many,
redirects` (previously `many, redirects`). It added up to +1.3 points of dev
Recall@30, mostly on code. The query that motivated it, `TooManyRedirects`, went
from not retrieved to rank 2. Its lowercased misspelling `toomanyredirects
exeption` went from not retrieved to rank 1.

### 4. Cross-encoder reranker over the pool's top 20

`cross-encoder/ms-marco-MiniLM-L6-v2` scores each (question, chunk) pair; the
best 5 go into the prompt (unchanged context budget).

| dev | Hit@1 | Hit@5 | MRR@5 |
|---|---|---|---|
| new pool, fused order | 44.2 | 79.6 | 57.7 |
| new pool, rerank top 20 | **54.9** | **81.4** | **64.4** |
| new pool, rerank top 30 | 54.9 | 80.5 | 63.9 |

Top 30 added nothing over top 20 and costs about 50% more time.

**Settings: max_length 512, batch 4.** On one thread (rag-qa has one vCPU),
20 pairs:

| max_length / batch | p50 | p95 | peak memory added |
|---|---|---|---|
| 512 / 32 | 1.40 s | 1.84 s | 304 MB |
| 512 / 8 | 1.15 s | 1.47 s | 184 MB |
| **512 / 4** | **1.04 s** | **1.28 s** | **138 MB** |
| 256 / 8 | 1.07 s | 1.19 s | 161 MB |

Small batches pad less, so they were both faster and lighter. Truncating to 256
tokens saved nothing over batch 4 and risked quality. Against rag-qa's
measured 1.17 GiB peak of 2 GiB, +138 MB leaves about 0.7 GiB of headroom.

**Time budget: 3 s.** Scoring is checked between batches; a query over budget,
or any model error, keeps the pool's fused order for that query and logs
`fallback=timeout` or `fallback=error`. In the final evaluation run, 0 of 181
queries fell back.

### 5. Model warm-up at startup

The service scales to zero. In a fresh process on one thread, the first
question took:

| | first question | next question |
|---|---|---|
| before (old pipeline; embedding model loaded on the first question) | 10.2 s | 0.03 s |
| new pipeline, models loaded on the first question | 9.8 s | 1.35 s |
| **new pipeline, models warmed in the background at page load** | **2.2 s** | 1.4 s |

The warm-up starts when the app first renders, in a background thread, and
took 7.9 s. The page itself renders in the same time with or without it
(10.7–11.3 s in a fresh test process, mostly imports). Readiness is
unaffected, because `/_stcore/health` does not run the app script. The
reranker alone loads in 7.4 s cold (most of that is importing
sentence-transformers). Its first scoring call is also slow: 3.2 s on the
first evaluation query against a 0.4 s median, enough to exceed the budget.
So the warm-up runs one dummy scoring after loading. CI loads and times the reranker inside the built image ("Check
the image").

### 6. README pin removed from the default path

The pin replaced the 5th result with a repository README chunk whenever the
README had any lexical match. On dev it never turned a repository-identity
miss into a hit: `rid-03` and `rid-06` were hits and `rid-01` and `rid-04` were
misses, with or without it. Meanwhile it fired on 86 of 162 answerable and 9 of
19 unanswerable queries. With reranking, it cost two queries their only gold
chunk (`isw-11`, `mh-01`) and gained none. It remains available as
`HybridRetriever(surface_readme=True)`.

### What did not improve, and open items

These are recorded in [`benchmark/v3/README.md`](../benchmark/v3/README.md):
repository identity (`rid-01`, `rid-04`), misspelled code queries, code
multi-hop, coverage of long lists, and a refusal threshold on the reranker
score (about −7.5; AUROC 0.886 on dev and 0.935 on holdout, not wired in).

## v4 (planned)

The plan for the next phases (sources display, a generation eval, typo
correction and intent, list and summary questions, answer length) is in
[`ROADMAP_v4.md`](ROADMAP_v4.md). Decisions are recorded here as each phase
lands. The retrieval holdout has been scored once (PR #6); the plan scores it
a second and final time in its last phase.

### Response cache keyed by chat, document set and question (2026-11-24)

The per-chat answer cache was keyed by the lowercased question alone. That
caused two concrete failures, each now covered by an `AppTest` that fails on
the old code:

* **New documents were ignored.** A question answered before a document was
  added returned the same answer, with the same sources, after it was added.
* **History was ignored.** A repeated follow-up ("and the next one?") returned
  its first answer even after the conversation had moved on.

**Now** (`src/response_cache.py`):

* **Key:** (chat id, document-set digest, normalized question). Normalizing
  means case, extra spaces and trailing `?!.` don't change the key.
* **Document-set digest:** each time a chat indexes documents or a
  repository, a SHA-256 over that batch's chunk sources and texts is
  recorded. The document set is the digest of those batch digests, sorted.
  Adding anything changes the key. Evicting the store resets it, along with
  the cache.
* **Follow-ups:** when the chat already has history, a message that leans on
  it is not read from or written to the cache. A message leans on history if
  any of these hold:
  * it has 3 words or fewer;
  * it opens with "and", "but", "what about", "why", "tell me more" and the
    like;
  * it contains a reference word: "it", "this", "they", "next", "previous",
    "second", "more", ...
* **Errs towards a cache miss.** "What does this library do?" also bypasses
  the cache. A wrong miss costs one Gemini call (about $0.001); a wrong hit
  answers a different question.
* **First messages are always cacheable.**

**Cost:** no new Gemini calls. A few hash operations per question.

### Phase 1: only the latest answer's sources in full (2026-11-24)

Before, every answer in a chat replayed its full "Sources:" list and chunk
expander, so a chat of ten questions showed ten source blocks.

**Now:**
* The latest answer shows its sources as before: the citation list, with the
  chunks in an expander.
* Each earlier answer keeps its **own** sources in one closed "Sources"
  expander directly beneath it. Sources were already stored on each message,
  so nothing about which sources belong to which answer changed.
* When a new question is sent, the previous answer collapses in the same run,
  not one rerun later. A question that cannot be answered (no key) adds no
  answer, so it collapses nothing.
* If the latest answer has no sources (a casual reply, an error), no block is
  shown in full.
* Streamlit forbids an expander inside an expander, so the collapsed version
  lists the chunks directly inside its single expander.

**Checks:**
* Four `AppTest` runs. Three of them fail on the previous code; the
  single-answer case looks the same before and after.
* A Playwright browser check (`scripts/ui_sources_check.js`, fake LLM):
  * after three questions there is exactly one full block (the latest);
  * the two earlier expanders start closed;
  * opening the first shows only its own file and chunk.

No retrieval, prompt or cache change; no latency or cost change.
