"""
Validate benchmark/queries.yaml against the real ingested corpus, resolve
span anchors, and run the vocabulary-overlap bias audit.

Two jobs, both of which must happen BEFORE any retrieval is measured:

1. Every locator must correspond to a file/page that actually produced chunks.
   A typo'd path would otherwise show up as a permanent retrieval failure and
   be misread as a system weakness.

2. The bias audit. `paraphrase` and `conceptual` questions are supposed to use
   different vocabulary from the text that answers them. If they do not, BM25
   scores them for free and the benchmark is rigged in favour of the hybrid
   arm. Overlap is measured with the project's own tokenizer, because that is
   the function whose behaviour actually decides BM25 candidacy.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

from benchmark import metrics
from benchmark.metrics import locator_of, normalise_locator  # noqa: F401 (re-export)
from src.hybrid_retrieval import tokenize

QUERIES_PATH = os.path.join(BENCHMARK_DIR, "queries.yaml")

# Overlap above this in a category that is supposed to paraphrase means the
# question probably inherited the source's wording. Not a hard threshold from
# theory - a review trigger.
SUSPICIOUS_OVERLAP = 0.60
PARAPHRASE_CATEGORIES = {"paraphrase", "conceptual", "doc_conceptual"}


# Locator identity lives in benchmark.metrics so the harness and this
# validator cannot drift apart on what "the same file/page" means.


def load_dataset(path: str = QUERIES_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# ── Span anchors ─────────────────────────────────────────────────────────────

def resolve_span(anchor: str, parents: dict, locators: list) -> dict | None:
    """
    Turn a span_anchor into a character range inside its parent document.

    Deliberately derived rather than hand-written: line numbers typed into
    YAML drift the moment the pin moves, and a silently stale span would
    corrupt the chunking diagnostic without failing anything.
    """
    wanted_files = {loc[0] for loc in locators}
    for key, parent in parents.items():
        # Match on filename only. A .docx parent is keyed (name, 0) while its
        # ground truth is expressed as ranges carrying no page, so an exact
        # key lookup would silently find nothing.
        if key[0] not in wanted_files:
            continue
        loc = key
        idx = parent.find(anchor)
        if idx >= 0:
            return {
                "locator": loc,
                "start": idx,
                "end": idx + len(anchor),
                "line": parent.count("\n", 0, idx) + 1,
            }
    return None


def parent_texts() -> dict:
    """{locator: full parent text} for every file in the corpus.

    Spans are character ranges inside the PARENT document, not inside a chunk,
    which is what lets the span metric survive a change to chunk size.
    """
    import io

    from benchmark import build_corpus
    from src.data_ingestion import load_documents
    from src.repo_ingestion import DEFAULT_LIMITS, read_archive

    parents: dict[tuple, str] = {}
    files, _ = read_archive(io.BytesIO(build_corpus.fetch_archive()),
                            limits=DEFAULT_LIMITS)
    for f in files:
        parents[(f.path, None)] = f.text
    for doc in load_documents(build_corpus.document_paths()):
        key = (os.path.basename(doc.metadata.get("source", "")),
               doc.metadata.get("page"))
        parents[key] = parents.get(key, "") + doc.page_content
    return parents


def attach_spans(queries: list, parents: dict | None = None) -> int:
    """Resolve every span_anchor to a character range, in place as q['_span'].

    Returns how many resolved. Raises if an anchor cannot be found, because a
    silently unresolved span would quietly shrink the span-metric denominator
    instead of failing.
    """
    parents = parents if parents is not None else parent_texts()
    resolved = 0
    for q in queries:
        anchor = q.get("span_anchor")
        if not anchor:
            continue
        locators = [normalise_locator(e) for e in q["relevant"]]
        span = resolve_span(anchor, parents, locators)
        if span is None:
            raise ValueError(f"{q['id']}: span_anchor {anchor!r} not found")
        q["_span"] = span
        resolved += 1
    return resolved


# ── Bias audit ───────────────────────────────────────────────────────────────

def overlap_fraction(question: str, text: str) -> float:
    """Share of the question's distinct tokens that literally occur in text."""
    q = set(tokenize(question))
    if not q:
        return 0.0
    return len(q & set(tokenize(text))) / len(q)


def describe(values: list) -> str:
    if not values:
        return "n=0"
    ordered = sorted(values)
    mid = ordered[len(ordered) // 2]
    return (f"n={len(ordered):2d}  min={ordered[0]:.2f}  "
            f"median={mid:.2f}  max={ordered[-1]:.2f}  "
            f"mean={sum(ordered)/len(ordered):.2f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    from benchmark.build_corpus import build_chunks

    dataset = load_dataset()
    queries = dataset["queries"]
    print(f"loaded {len(queries)} queries from queries.yaml\n")

    chunks, _meta = build_chunks()

    # chunks grouped by locator, and parent text per locator for span anchors
    by_locator: dict[tuple, list] = defaultdict(list)
    for chunk in chunks:
        by_locator[locator_of(chunk)].append(chunk)

    parents = parent_texts()

    errors: list[str] = []
    warnings: list[str] = []
    spec_chunk_counts: dict = defaultdict(list)
    per_category: dict[str, list] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    resolved_spans = 0

    # ── 1. structural + locator validation ──────────────────────────────────
    seen_ids = set()
    for q in queries:
        qid, cat = q.get("id"), q.get("category")
        counts[cat] += 1
        if qid in seen_ids:
            errors.append(f"{qid}: duplicate id")
        seen_ids.add(qid)
        for field in ("id", "category", "question", "relevant", "notes"):
            if field not in q:
                errors.append(f"{qid}: missing field {field!r}")

        specs = [metrics.normalise_spec(e) for e in q["relevant"]]
        locators = [(s[1], None) for s in specs]

        if cat == "unanswerable":
            if specs:
                errors.append(f"{qid}: unanswerable must have empty `relevant`")
        elif not specs:
            errors.append(f"{qid}: no locators")

        # Every spec must actually resolve to chunks in the freshly built
        # corpus. A typo'd path or a stale character range would otherwise
        # look like a permanent retrieval failure.
        for spec in specs:
            n_matching = sum(1 for c in chunks if metrics.chunk_matches(spec, c))
            if n_matching == 0:
                errors.append(f"{qid}: ground-truth spec {spec} matches NO chunk")
            spec_chunk_counts[qid].append((spec, n_matching))

        if cat == "cross_file" and len(locators) < 2:
            errors.append(f"{qid}: cross_file needs >= 2 locators, has {len(locators)}")

        # span anchors
        anchor = q.get("span_anchor")
        if anchor:
            span = resolve_span(anchor, parents, locators)
            if span is None:
                errors.append(f"{qid}: span_anchor {anchor!r} not found in any parent")
            else:
                q["_span"] = span
                resolved_spans += 1

    # ── 2. bias audit ───────────────────────────────────────────────────────
    all_texts = [c.page_content for c in chunks]
    rows = []
    for q in queries:
        qid, cat, question = q["id"], q["category"], q["question"]

        if cat == "unanswerable":
            # No ground truth. Best lexical match anywhere in the corpus tells
            # us how plausible the distractor is - low is good here.
            best = max((overlap_fraction(question, t) for t in all_texts), default=0.0)
            rows.append((qid, cat, best, "best-in-corpus"))
            per_category[cat].append(best)
            continue

        specs = [metrics.normalise_spec(e) for e in q["relevant"]]
        gt_chunks = [c for c in chunks
                     if any(metrics.chunk_matches(s, c) for s in specs)]
        best = max((overlap_fraction(question, c.page_content) for c in gt_chunks),
                   default=0.0)
        rows.append((qid, cat, best, "vs ground truth"))
        per_category[cat].append(best)
        if cat in PARAPHRASE_CATEGORIES and best >= SUSPICIOUS_OVERLAP:
            warnings.append(
                f"{qid} [{cat}] overlap {best:.2f} >= {SUSPICIOUS_OVERLAP:.2f} "
                f"- question may have inherited source vocabulary: {question[:70]}"
            )

    # ── report ──────────────────────────────────────────────────────────────
    print("category counts:")
    for cat in sorted(counts):
        print(f"   {cat:18} {counts[cat]:3d}")
    print(f"   {'TOTAL':18} {sum(counts.values()):3d}")
    print(f"\nspan anchors resolved: {resolved_spans}")

    print("\n" + "=" * 74)
    print("BIAS AUDIT - question/answer token overlap, project tokenizer")
    print("=" * 74)
    print("fraction of the question's distinct tokens that literally appear in")
    print("its ground-truth text. High is EXPECTED for exact_identifier and")
    print("literal_string; high for paraphrase/conceptual means contamination.\n")
    for cat in sorted(per_category):
        print(f"  {cat:18} {describe(per_category[cat])}")

    print("\nper-query:")
    for qid, cat, val, kind in rows:
        flag = ""
        if cat in PARAPHRASE_CATEGORIES and val >= SUSPICIOUS_OVERLAP:
            flag = "  <-- SUSPICIOUS"
        print(f"  {qid:8} {cat:18} {val:.2f}  ({kind}){flag}")

    if warnings:
        print("\n" + "!" * 74)
        print(f"{len(warnings)} SUSPICIOUS QUERY(S) - rewrite before measuring:")
        for w in warnings:
            print("  " + w)
        print("!" * 74)

    if errors:
        print("\n" + "X" * 74)
        print(f"{len(errors)} VALIDATION ERROR(S):")
        for e in errors:
            print("  " + e)
        print("X" * 74)
        return 1

    print("\nAll locators verified against the ingested corpus. No errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
