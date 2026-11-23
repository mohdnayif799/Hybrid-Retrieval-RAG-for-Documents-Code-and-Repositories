"""
Span-level ground truth for the v3 benchmark.

A query's ``gold`` is a list of spans. Each span names a character range
inside one parent text (a repository file, a PDF page, the .txt or the .docx):

    - file: src/requests/models.py         # repo path, or document basename
      anchor: "DEFAULT_REDIRECT_LIMIT: int = 30"
      evidence: "30"                       # must occur inside the span
      occurrence: 1                        # only when the anchor is not unique
      part: limit                          # spans sharing a part are alternatives
      page: 0                              # only for multi-page PDFs

    - file: V_SEM_3_to_5_units_ETCE.docx   # a section: from `anchor` to the end
      anchor: "Steps Involved in Report Writing"   # of the first `until` after it
      until: "7. Format and Finalize"
      evidence: "Edit and Revise"

    - file: V_SEM_3_to_5_units_ETCE.docx   # legacy v2 range form
      start: 6155
      end: 7253

Anchors, not typed offsets: an anchor is located in the real parent text at
load time, so a span cannot silently drift. An anchor must occur exactly once
in its parent unless ``occurrence`` says which copy is meant; a non-unique
anchor without it is an error, because it would make the span ambiguous.

``evidence`` is the answer itself (a value, a name, a key phrase). Validation
checks that it actually occurs inside the resolved span, compared
case-insensitively with whitespace collapsed. That is the check that the
labelled location really holds the answer, not just a related heading.

Parts. A query may need several pieces of information (a multi-hop question)
or accept any of several equivalent locations (the same fact stated in the
docs and in the code). Spans with the same ``part`` are alternatives for one
piece; different parts are different pieces. Hit@k asks whether ANY part is
covered in the top k; Recall@k is the share of parts covered. Spans without a
``part`` each form their own part. ``part`` may also be a list, for one span
that answers several pieces at once (a docs paragraph covering both halves of
a two-part question).

Near-duplicates. A chunk that holds a copy of a gold chunk's text also counts
as gold, for that gold chunk's parts: retrieving a verbatim copy of the answer
(the .docx's duplicated block, a docstring repeated on two methods) is not a
retrieval failure. The test is DIRECTIONAL: candidate C counts for gold chunk
G when C contains at least 80% of G's 5-word shingles and G has at least 10 of
them. Two weaker tests were tried on this corpus and rejected:
  * symmetric containment (measured against the smaller chunk) admitted a
    12-word boilerplate fragment ("This method should not be called from user
    code...") as a copy of every docstring that contains it;
  * embedding cosine > 0.9 admitted a deprecated sibling method's docstring
    that mentions, but does not define, the gold identifier.
"""

from __future__ import annotations

import ntpath
import re
from dataclasses import dataclass

CONTAINMENT_THRESHOLD = 0.8
MIN_GOLD_SHINGLES = 10
SHINGLE_WORDS = 5


@dataclass(frozen=True)
class Span:
    file: str
    page: int
    start: int
    end: int
    parts: tuple[str, ...]
    anchor: str | None = None
    evidence: str | None = None


class GoldError(ValueError):
    pass


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _parents_for(parents: dict, file: str, page):
    keys = [k for k in parents if k[0] == file and (page is None or k[1] == page)]
    if not keys:
        raise GoldError(f"no parent text for file={file!r} page={page!r}")
    return sorted(keys, key=lambda k: (k[1] is None, k[1]))


def resolve_span(entry: dict, parents: dict, default_part: str) -> Span:
    """One YAML gold entry -> a Span with a character range in its parent."""
    file = entry.get("file")
    if not file:
        raise GoldError(f"gold entry without file: {entry!r}")
    page = entry.get("page")
    raw_part = entry.get("part", default_part)
    parts = tuple(str(p) for p in raw_part) if isinstance(raw_part, list) else (str(raw_part),)
    if not parts:
        raise GoldError(f"empty part list in {entry!r}")
    evidence = entry.get("evidence")

    if "anchor" in entry:
        anchor = entry["anchor"]
        if not anchor or not anchor.strip():
            raise GoldError(f"empty anchor in {entry!r}")
        hits = []
        for key in _parents_for(parents, file, page):
            text = parents[key]
            idx = text.find(anchor)
            while idx >= 0:
                hits.append((key, idx))
                idx = text.find(anchor, idx + 1)
        if not hits:
            raise GoldError(f"anchor not found in {file}: {anchor!r}")
        occurrence = entry.get("occurrence")
        if occurrence is None:
            if len(hits) > 1:
                raise GoldError(
                    f"anchor occurs {len(hits)} times in {file}; give `occurrence`: {anchor!r}")
            occurrence = 1
        if not 1 <= int(occurrence) <= len(hits):
            raise GoldError(f"occurrence {occurrence} out of range ({len(hits)}) for {anchor!r}")
        (fkey, pg), start = hits[int(occurrence) - 1]
        end = start + len(anchor)
        until = entry.get("until")
        if until:
            stop = parents[(fkey, pg)].find(until, start + 1)
            if stop < 0:
                raise GoldError(f"`until` {until!r} not found after anchor {anchor!r} in {file}")
            end = stop + len(until)
        span = Span(file=fkey, page=pg, start=start, end=end, parts=parts,
                    anchor=anchor, evidence=evidence)
    elif "start" in entry and "end" in entry:
        keys = _parents_for(parents, file, page)
        if len(keys) != 1:
            raise GoldError(f"range span on multi-page file needs `page`: {entry!r}")
        start, end = int(entry["start"]), int(entry["end"])
        if not 0 <= start < end <= len(parents[keys[0]]):
            raise GoldError(f"range [{start},{end}) outside {file} ({len(parents[keys[0]])} chars)")
        span = Span(file=file, page=keys[0][1], start=start, end=end, parts=parts,
                    evidence=evidence)
    else:
        raise GoldError(f"gold entry needs `anchor` or `start`/`end`: {entry!r}")

    if evidence is not None:
        text = parent_slice(parents, span)
        if _norm(evidence) not in _norm(text):
            raise GoldError(
                f"evidence {evidence!r} not inside span {span.file}[{span.start}:{span.end}]")
    return span


def parent_slice(parents: dict, span: Span) -> str:
    return parents[(span.file, span.page)][span.start:span.end]


def resolve_gold(query: dict, parents: dict) -> list[Span]:
    return [resolve_span(e, parents, default_part=f"p{i}")
            for i, e in enumerate(query.get("gold") or [])]


def n_parts(spans: list[Span]) -> int:
    return len({p for s in spans for p in s.parts})


# ── chunk matching ────────────────────────────────────────────────────────────

def chunk_file(chunk) -> str:
    source = chunk.metadata.get("source", "")
    if chunk.metadata.get("source_type") == "repo":
        return source
    return ntpath.basename(source)


def chunk_overlaps(span: Span, chunk) -> bool:
    if chunk_file(chunk) != span.file:
        return False
    if chunk.metadata.get("page", 0) != span.page:
        return False
    begin = chunk.metadata.get("start_index")
    if begin is None:
        return False
    return begin < span.end and span.start < begin + len(chunk.page_content)


def direct_gold(spans: list[Span], chunks: list) -> dict[int, set[str]]:
    """chunk index -> parts it covers, by direct overlap with a gold span."""
    out: dict[int, set[str]] = {}
    for i, chunk in enumerate(chunks):
        parts = {p for s in spans if chunk_overlaps(s, chunk) for p in s.parts}
        if parts:
            out[i] = parts
    return out


def shingles(text: str, n: int = SHINGLE_WORDS) -> frozenset:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[k:k + n]) for k in range(len(words) - n + 1))


def contained_share(gold: frozenset, candidate: frozenset) -> float:
    """Share of the gold chunk's shingles that also occur in the candidate."""
    if not gold:
        return 0.0
    return len(gold & candidate) / len(gold)


class NearDuplicates:
    """
    For every chunk g, the chunks that contain (almost all of) g's text.

    ``copies_of[g]`` is directional: c is listed when c holds at least
    ``threshold`` of g's shingles. Used only to widen gold, never to narrow it.
    """

    def __init__(self, chunks: list, threshold: float = CONTAINMENT_THRESHOLD,
                 min_shingles: int = MIN_GOLD_SHINGLES):
        self.sh = [shingles(c.page_content) for c in chunks]
        self.copies_of: dict[int, set[int]] = {i: set() for i in range(len(chunks))}
        index: dict[str, set[int]] = {}
        for i, s in enumerate(self.sh):
            for g in s:
                index.setdefault(g, set()).add(i)
        for g_idx, g_sh in enumerate(self.sh):
            if len(g_sh) < min_shingles:
                continue
            counts: dict[int, int] = {}
            for shingle in g_sh:
                for c in index[shingle]:
                    if c != g_idx:
                        counts[c] = counts.get(c, 0) + 1
            need = threshold * len(g_sh)
            self.copies_of[g_idx] = {c for c, n in counts.items() if n >= need}


def expand_gold(direct: dict[int, set[str]], near: NearDuplicates | None) -> dict[int, set[str]]:
    """Direct gold plus near-duplicates of gold chunks, carrying the same parts."""
    out = {i: set(p) for i, p in direct.items()}
    if near is None:
        return out
    for i, parts in direct.items():
        for j in near.copies_of.get(i, ()):
            out.setdefault(j, set()).update(parts)
    return out


# ── rank metrics over chunk-index lists ───────────────────────────────────────

def first_rank(ranked: list[int], gold: dict[int, set[str]]) -> int | None:
    for r, i in enumerate(ranked, start=1):
        if i in gold:
            return r
    return None


def hit_at(ranked, gold, k) -> bool:
    r = first_rank(ranked[:k], gold)
    return r is not None


def mrr_at(ranked, gold, k=5) -> float:
    r = first_rank(ranked[:k], gold)
    return 1.0 / r if r else 0.0


def recall_at(ranked, gold, n_parts: int, k=20) -> float:
    if n_parts <= 0:
        raise ValueError("recall is undefined without gold parts")
    covered: set[str] = set()
    for i in ranked[:k]:
        covered |= gold.get(i, set())
    return len(covered) / n_parts
