"""
Load and validate the v3 query set (benchmark/v3/queries.yaml).

Every query carries:
  id, category, qtype, domain, question, notes, origin (v2 | v3)
  gold       span-level ground truth (see benchmark/v3/gold.py); empty for
             unanswerable queries
  relevant   v2's file-level locators, kept for the secondary file-level
             metric on code queries

``validate`` resolves every gold span against the real corpus text, checks
each span's evidence, and runs the vocabulary-overlap audit. It never edits the
file; problems come back as errors and warnings.

    python -m benchmark.v3.dataset [--archive PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

V3_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(V3_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

from benchmark.v3 import gold as G

QUERIES_PATH = os.path.join(V3_DIR, "queries.yaml")

QTYPES = ("exact_identifier", "keyword", "paraphrase", "multi_hop",
          "misspelled", "list_all", "repo_identity", "unanswerable")
DOMAINS = ("code", "docs")
ORIGINS = ("v2", "v3")
REQUIRED = ("id", "category", "qtype", "domain", "question", "notes", "origin")

# Overlap of a question's tokens with its gold text above this, in a type that
# is supposed to reword the source, means the question probably borrowed the
# source's vocabulary - which hands BM25 the answer. A review trigger, not a rule.
SUSPICIOUS_OVERLAP = 0.60
REWORDED_QTYPES = {"paraphrase", "multi_hop"}


def load(path: str = QUERIES_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def answerable(queries: list) -> list:
    return [q for q in queries if q["qtype"] != "unanswerable"]


def overlap_fraction(question: str, text: str) -> float:
    """Share of the question's distinct project tokens that occur in text."""
    from src.hybrid_retrieval import tokenize

    q = set(tokenize(question))
    if not q:
        return 0.0
    return len(q & set(tokenize(text))) / len(q)


@dataclass
class Resolved:
    query: dict
    spans: list = field(default_factory=list)
    direct: dict = field(default_factory=dict)      # chunk index -> parts
    expanded: dict = field(default_factory=dict)    # incl. near-duplicates
    n_parts: int = 0
    overlap: float = 0.0


def resolve_all(queries: list, chunks: list, parents: dict,
                near: G.NearDuplicates | None):
    """Resolve gold for every answerable query. Returns ({id: Resolved}, errors)."""
    out: dict[str, Resolved] = {}
    errors: list[str] = []
    for q in queries:
        r = Resolved(query=q)
        out[q["id"]] = r
        if q["qtype"] == "unanswerable":
            continue
        try:
            r.spans = G.resolve_gold(q, parents)
        except G.GoldError as exc:
            errors.append(f"{q['id']}: {exc}")
            continue
        r.n_parts = G.n_parts(r.spans)
        r.direct = G.direct_gold(r.spans, chunks)
        r.expanded = G.expand_gold(r.direct, near)
        missing = {p for s in r.spans for p in s.parts} - {p for v in r.direct.values() for p in v}
        if missing:
            errors.append(f"{q['id']}: gold part(s) {sorted(missing)} overlap no chunk")
        text = " ".join(G.parent_slice(parents, s) for s in r.spans)
        r.overlap = overlap_fraction(q["question"], text)
    return out, errors


def validate(data: dict, chunks: list, parents: dict,
             near: G.NearDuplicates | None = None):
    """Structural checks plus span resolution. Returns (resolved, errors, warnings)."""
    queries = data["queries"]
    errors: list[str] = []
    warnings: list[str] = []
    seen = set()
    for q in queries:
        qid = q.get("id")
        for key in REQUIRED:
            if key not in q:
                errors.append(f"{qid}: missing field {key!r}")
        if qid in seen:
            errors.append(f"{qid}: duplicate id")
        seen.add(qid)
        if q.get("qtype") not in QTYPES:
            errors.append(f"{qid}: unknown qtype {q.get('qtype')!r}")
        if q.get("domain") not in DOMAINS:
            errors.append(f"{qid}: unknown domain {q.get('domain')!r}")
        if q.get("origin") not in ORIGINS:
            errors.append(f"{qid}: unknown origin {q.get('origin')!r}")
        has_gold = bool(q.get("gold"))
        if q.get("qtype") == "unanswerable" and has_gold:
            errors.append(f"{qid}: unanswerable query must not have gold")
        if q.get("qtype") != "unanswerable" and not has_gold:
            errors.append(f"{qid}: answerable query has no gold")
        if q.get("origin") == "v3":
            for entry in q.get("gold") or []:
                if not entry.get("evidence"):
                    errors.append(f"{qid}: new queries need `evidence` on every gold span")
    resolved, span_errors = resolve_all(queries, chunks, parents, near)
    errors.extend(span_errors)
    for qid, r in resolved.items():
        if r.query.get("qtype") in REWORDED_QTYPES and r.overlap >= SUSPICIOUS_OVERLAP:
            warnings.append(f"{qid} [{r.query['qtype']}] question/gold token overlap "
                            f"{r.overlap:.2f} >= {SUSPICIOUS_OVERLAP}")
    return resolved, errors, warnings


def summary_lines(data: dict, resolved: dict) -> list[str]:
    queries = data["queries"]
    lines = [f"{len(queries)} queries, {len(answerable(queries))} answerable"]
    by = Counter((q["domain"], q["qtype"], q["origin"]) for q in queries)
    for (dom, qt, org), n in sorted(by.items()):
        lines.append(f"  {dom:5} {qt:17} {org}  {n:3d}")
    gold_sizes = defaultdict(list)
    for r in resolved.values():
        if r.query["qtype"] != "unanswerable" and r.direct:
            gold_sizes[r.query["domain"]].append(len(r.direct))
    for dom, sizes in sorted(gold_sizes.items()):
        s = sorted(sizes)
        lines.append(f"  gold chunks per query [{dom}]: min {s[0]}  median {s[len(s)//2]}  max {s[-1]}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", help="local psf/requests tarball at the pinned commit")
    args = parser.parse_args()

    from benchmark.v3.corpus import load_corpus

    chunks, parents, _meta = load_corpus(args.archive)
    data = load()
    near = G.NearDuplicates(chunks)
    resolved, errors, warnings = validate(data, chunks, parents, near)
    for line in summary_lines(data, resolved):
        print(line)
    for w in warnings:
        print("WARNING", w)
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
