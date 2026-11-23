"""
Tests for the v3 retrieval benchmark (benchmark/v3).

Synthetic fixtures only, plus structural checks on the committed query file.
Measured retrieval numbers are measurements, not invariants, and are never
asserted on. The one exception is freshness: committed v3 results must have
been produced by the code that is committed next to them.
"""

from __future__ import annotations

import json
import os
from typing import ClassVar

import numpy as np
import pytest
from langchain_core.documents import Document

from benchmark.v3 import dataset as D
from benchmark.v3 import gold as G
from benchmark.v3 import stats as S
from benchmark.v3.corpus import content_digest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARENT = (
    "alpha line one\n"
    "def target_function(x):\n"
    "    return x * 42\n"
    "beta line two\n"
    "def target_function(y):\n"
    "    return y\n"
    "SECTION START\nitem one\nitem two\nSECTION END\n"
)
PARENTS = {("pkg/mod.py", 0): PARENT, ("notes.pdf", 0): "page zero text", ("notes.pdf", 1): "page one answer"}


def chunk(source, start, text, repo=True, page=0):
    meta = {"source": source, "start_index": start, "page": page}
    if repo:
        meta["source_type"] = "repo"
    return Document(page_content=text, metadata=meta)


# ── gold resolution ──────────────────────────────────────────────────────────

class TestResolveSpan:
    def test_unique_anchor(self):
        s = G.resolve_span({"file": "pkg/mod.py", "anchor": "return x * 42", "evidence": "42"},
                           PARENTS, "p0")
        assert PARENT[s.start:s.end] == "return x * 42"
        assert s.parts == ("p0",)

    def test_non_unique_anchor_needs_occurrence(self):
        with pytest.raises(G.GoldError, match="occurs 2 times"):
            G.resolve_span({"file": "pkg/mod.py", "anchor": "def target_function("}, PARENTS, "p0")

    def test_occurrence_selects_copy(self):
        s = G.resolve_span({"file": "pkg/mod.py", "anchor": "def target_function(",
                            "occurrence": 2}, PARENTS, "p0")
        assert s.start == PARENT.index("def target_function(y)")

    def test_until_extends_span(self):
        s = G.resolve_span({"file": "pkg/mod.py", "anchor": "SECTION START", "until": "SECTION END",
                            "evidence": "item two"}, PARENTS, "p0")
        assert PARENT[s.start:s.end].endswith("SECTION END")

    def test_evidence_must_be_inside_span(self):
        with pytest.raises(G.GoldError, match="evidence"):
            G.resolve_span({"file": "pkg/mod.py", "anchor": "alpha line one",
                            "evidence": "beta"}, PARENTS, "p0")

    def test_evidence_ignores_case_and_whitespace(self):
        G.resolve_span({"file": "pkg/mod.py", "anchor": "SECTION START", "until": "SECTION END",
                        "evidence": "ITEM   ONE item two"}, PARENTS, "p0")

    def test_missing_anchor(self):
        with pytest.raises(G.GoldError, match="not found"):
            G.resolve_span({"file": "pkg/mod.py", "anchor": "nope"}, PARENTS, "p0")

    def test_range_bounds_checked(self):
        with pytest.raises(G.GoldError, match="outside"):
            G.resolve_span({"file": "pkg/mod.py", "start": 0, "end": 10_000}, PARENTS, "p0")

    def test_page_scoping(self):
        s = G.resolve_span({"file": "notes.pdf", "anchor": "answer"}, PARENTS, "p0")
        assert s.page == 1

    def test_part_list(self):
        s = G.resolve_span({"file": "pkg/mod.py", "anchor": "alpha line one", "part": ["a", "b"]},
                           PARENTS, "p0")
        assert s.parts == ("a", "b")
        assert G.n_parts([s]) == 2


# ── chunk matching, near-duplicates, metrics ────────────────────────────────

class TestChunkGold:
    def setup_method(self):
        self.span = G.Span(file="pkg/mod.py", page=0, start=10, end=20, parts=("a",))

    def test_overlap(self):
        assert G.chunk_overlaps(self.span, chunk("pkg/mod.py", 15, "x" * 3))
        assert not G.chunk_overlaps(self.span, chunk("pkg/mod.py", 20, "x" * 3))
        assert not G.chunk_overlaps(self.span, chunk("other.py", 15, "x" * 3))
        assert not G.chunk_overlaps(self.span, chunk("pkg/mod.py", 15, "x", page=1))

    def test_document_chunks_match_on_basename(self):
        span = G.Span(file="notes.pdf", page=1, start=0, end=5, parts=("a",))
        assert G.chunk_overlaps(span, chunk("/abs/path/notes.pdf", 0, "page", repo=False, page=1))

    def test_direct_gold(self):
        chunks = [chunk("pkg/mod.py", 0, "a" * 12), chunk("pkg/mod.py", 12, "b" * 12)]
        assert G.direct_gold([self.span], chunks) == {0: {"a"}, 1: {"a"}}


WORDS = " ".join(f"w{i}" for i in range(40))


class TestNearDuplicates:
    def test_full_copy_counts(self):
        chunks = [chunk("a.py", 0, WORDS), chunk("b.py", 0, "prefix " + WORDS)]
        near = G.NearDuplicates(chunks)
        assert 1 in near.copies_of[0]

    def test_fragment_is_not_a_copy_of_the_whole(self):
        fragment = " ".join(WORDS.split()[:14])
        chunks = [chunk("a.py", 0, WORDS), chunk("b.py", 0, fragment)]
        near = G.NearDuplicates(chunks)
        assert 1 not in near.copies_of[0]       # fragment holds < 80% of the gold chunk
        assert near.copies_of[1] == {0}          # the whole does contain the fragment

    def test_short_gold_chunks_are_not_expanded(self):
        chunks = [chunk("a.py", 0, "one two three"), chunk("b.py", 0, "one two three four")]
        assert G.NearDuplicates(chunks).copies_of[0] == set()

    def test_expand_carries_parts(self):
        chunks = [chunk("a.py", 0, WORDS), chunk("b.py", 0, WORDS)]
        out = G.expand_gold({0: {"x"}}, G.NearDuplicates(chunks))
        assert out == {0: {"x"}, 1: {"x"}}


class TestRankMetrics:
    gold: ClassVar[dict] = {3: {"a"}, 7: {"b"}}

    def test_rank_hit_mrr(self):
        assert G.first_rank([1, 3, 7], self.gold) == 2
        assert G.hit_at([1, 3], self.gold, 1) is False
        assert G.mrr_at([1, 3], self.gold, 5) == 0.5
        assert G.mrr_at([1, 2], self.gold, 5) == 0.0

    def test_recall_counts_parts_not_chunks(self):
        assert G.recall_at([3, 3, 1], self.gold, n_parts=2, k=20) == 0.5
        assert G.recall_at([3, 7], self.gold, n_parts=2, k=20) == 1.0
        assert G.recall_at([1, 3, 7], self.gold, n_parts=2, k=2) == 0.5

    def test_recall_undefined_without_parts(self):
        with pytest.raises(ValueError):
            G.recall_at([1], {}, n_parts=0)


# ── bootstrap ────────────────────────────────────────────────────────────────

class TestBootstrap:
    def test_deterministic(self):
        a = S.resample_matrix(30, 500, seed=1)
        b = S.resample_matrix(30, 500, seed=1)
        assert np.array_equal(a, b)

    def test_interval_contains_mean(self):
        values = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
        out = S.mean_ci(values, S.resample_matrix(len(values), 2000))
        assert out["ci_low"] <= out["mean"] <= out["ci_high"]
        assert out["mean"] == pytest.approx(0.7)

    def test_identical_arms_have_zero_difference(self):
        v = [1, 0, 1, 0, 1]
        out = S.paired_diff_ci(v, v, S.resample_matrix(5, 200))
        assert out["diff"] == out["ci_low"] == out["ci_high"] == 0.0
        assert out["excludes_zero"] is False

    def test_consistent_difference_excludes_zero(self):
        a, b = [1] * 40, [0] * 40
        assert S.paired_diff_ci(a, b, S.resample_matrix(40, 500))["excludes_zero"] is True

    def test_shape_checks(self):
        with pytest.raises(ValueError):
            S.mean_ci([1, 0], S.resample_matrix(3, 10))
        with pytest.raises(ValueError):
            S.paired_diff_ci([1, 0], [1], S.resample_matrix(2, 10))
        with pytest.raises(ValueError):
            S.resample_matrix(0)


# ── corpus digest ────────────────────────────────────────────────────────────

class _F:
    def __init__(self, path, text):
        self.path, self.text = path, text


def test_content_digest_is_order_independent_and_content_sensitive():
    a = [_F("x.py", "1"), _F("y.py", "2")]
    assert content_digest(a) == content_digest(list(reversed(a)))
    assert content_digest(a) != content_digest([_F("x.py", "1"), _F("y.py", "3")])


# ── dataset validation (synthetic) ───────────────────────────────────────────

def _q(**kw):
    base = {"id": "q1", "category": "c", "qtype": "keyword", "domain": "code", "origin": "v3",
            "question": "target function", "notes": "n",
            "gold": [{"file": "pkg/mod.py", "anchor": "return x * 42", "evidence": "42"}]}
    base.update(kw)
    return base


class TestValidate:
    chunks: ClassVar[list] = [chunk("pkg/mod.py", 0, PARENT[:60]),
                              chunk("pkg/mod.py", 60, PARENT[60:])]

    def run(self, *queries):
        return D.validate({"queries": list(queries)}, self.chunks, PARENTS)

    def test_clean(self):
        _, errors, _ = self.run(_q())
        assert errors == []

    def test_unanswerable_with_gold_rejected(self):
        _, errors, _ = self.run(_q(qtype="unanswerable"))
        assert any("must not have gold" in e for e in errors)

    def test_answerable_without_gold_rejected(self):
        _, errors, _ = self.run(_q(gold=[]))
        assert any("no gold" in e for e in errors)

    def test_new_query_needs_evidence(self):
        _, errors, _ = self.run(_q(gold=[{"file": "pkg/mod.py", "anchor": "return x * 42"}]))
        assert any("evidence" in e for e in errors)

    def test_duplicate_id_and_bad_enums(self):
        _, errors, _ = self.run(_q(), _q(qtype="bogus", domain="x", origin="v9"))
        joined = " ".join(errors)
        assert "duplicate id" in joined and "qtype" in joined and "domain" in joined

    def test_overlap_warning_for_reworded_types(self):
        _, _, warnings = self.run(_q(qtype="paraphrase", question="return x"))
        assert warnings


# ── the committed query file (structure only; spans need the corpus) ────────

def test_committed_query_file_structure():
    data = D.load()
    queries = data["queries"]
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids))
    for q in queries:
        for key in D.REQUIRED:
            assert key in q, (q.get("id"), key)
        assert q["qtype"] in D.QTYPES and q["domain"] in D.DOMAINS and q["origin"] in D.ORIGINS
        if q["qtype"] == "unanswerable":
            assert not q.get("gold"), q["id"]
        else:
            assert q.get("gold"), q["id"]
        if q["origin"] == "v3":
            assert all(e.get("evidence") for e in q.get("gold") or []), q["id"]


# ── harness end to end on a tiny corpus with a fake embedder ────────────────

def test_harness_runs_production_path_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    pytest.importorskip("rank_bm25")
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from benchmark.v3 import evaluate as E
    from src import vector_store

    monkeypatch.setattr(vector_store, "get_embeddings", lambda: DeterministicFakeEmbedding(size=16))
    texts = ["def parse_header(value): split the header on commas",
             "README project overview: a tiny HTTP helper",
             "def retry_policy(): retries failed connections",
             "unrelated text about cooking pasta"]
    sources = ["pkg/headers.py", "README.md", "pkg/retry.py", "pkg/misc.py"]
    chunks = [chunk(s, 0, t) for s, t in zip(sources, texts)]
    parents = {(s, 0): t for s, t in zip(sources, texts)}
    data = {"queries": [
        {"id": "a1", "category": "c", "qtype": "keyword", "domain": "code", "origin": "v3",
         "question": "parse header", "notes": "n", "relevant": ["pkg/headers.py"],
         "gold": [{"file": "pkg/headers.py", "anchor": "def parse_header", "evidence": "parse_header"}]},
        {"id": "u1", "category": "unanswerable", "qtype": "unanswerable", "domain": "code",
         "origin": "v3", "question": "websocket", "notes": "n"},
    ]}
    store = E.build_store(chunks, str(tmp_path / "store"), exact=True)
    assert E.store_hnsw_config(store)["ef_search"] == E.EXACT_HNSW["ef_search"]
    result = E.run(chunks, parents, data, store, splits={})
    row = result["per_query"]["a1"]
    assert set(row["arms"]) == set(E.ARMS)
    for arm in E.ARMS:
        for metric in ("hit@1", "hit@5", "mrr@5", "recall@20"):
            assert 0.0 <= row["arms"][arm][metric] <= 1.0
    for metric in ("all_parts@5", "chunk_cov@5", "chunk_cov@20"):
        assert 0.0 <= row["arms"]["production"][metric] <= 1.0
    unans = result["per_query"]["u1"]
    assert unans["arms"] == {}
    for arm in E.ARMS:                      # refusal signals recorded for every query
        for r in (unans, row):
            cosine = r["top1"][arm]["cosine"]
            assert cosine is None or -1.01 <= cosine <= 1.01   # None: the arm returned nothing
    assert unans["top1"]["dense_plain"]["cosine"] is not None   # dense always returns chunks
    assert unans["top1"]["bm25"]["cosine"] is None              # no shared token, no candidates
    assert row["top1"]["production"]["native"] is not None   # RRF score
    agg = E.aggregate(result["per_query"], resamples=50, seed=1)
    assert agg["all/all/all"]["n"] == 1
    signal = E.refusal_signal(result["per_query"])["all"]["production"]
    assert signal["unanswerable_cosine"]["n"] == 1 and signal["answerable_cosine"]["n"] == 1


def test_scoring_refuses_unfrozen_set_without_flag(monkeypatch, tmp_path):
    from benchmark.v3 import evaluate as E

    monkeypatch.setattr(E, "MANIFEST_PATH", str(tmp_path / "missing.json"))
    with pytest.raises(SystemExit, match="not frozen"):
        E.load_splits(allow_unfrozen=False)
    assert E.load_splits(allow_unfrozen=True) == ({}, None)


# ── freshness of committed results ──────────────────────────────────────────

def test_committed_v3_results_match_the_code():
    """Committed results must come from the committed retrieval code and query set."""
    from benchmark.v3 import evaluate as E

    if not os.path.exists(E.RESULTS_PATH):
        pytest.skip("no committed v3 results yet")
    with open(E.RESULTS_PATH, encoding="utf-8") as handle:
        recorded = json.load(handle)["fingerprint"]
    stale = [path for path, digest in E.fingerprint().items() if recorded.get(path) != digest]
    assert not stale, (
        f"benchmark/v3/results.json is stale for {stale}: re-run "
        "`python -m benchmark.v3.evaluate` and commit the new results")


# ── new metrics ──────────────────────────────────────────────────────────────

class TestNewMetrics:
    def test_chunk_coverage(self):
        from benchmark.v3.evaluate import chunk_coverage

        direct = {1: {"a"}, 2: {"a"}, 3: {"a"}, 4: {"a"}}
        assert chunk_coverage([1, 9, 2], direct, 5) == 0.5
        assert chunk_coverage([9, 9, 9, 9, 9, 1, 2, 3, 4], direct, 5) == 0.0
        assert chunk_coverage([9, 9, 9, 9, 9, 1, 2, 3, 4], direct, 20) == 1.0
        with pytest.raises(ValueError):
            chunk_coverage([1], {}, 5)

    def test_all_parts_needs_every_part(self):
        from benchmark.v3.evaluate import score_query

        gold = {1: {"a"}, 2: {"b"}}
        one = score_query([1, 9, 9], [1, 9, 9], gold, 2, gold)
        both = score_query([9, 2, 1], [9, 2, 1], gold, 2, gold)
        assert one["hit@5"] == 1.0 and one["all_parts@5"] == 0.0
        assert both["all_parts@5"] == 1.0 and both["mrr@5"] == 0.5

    def test_group_tags(self):
        from benchmark.v3.evaluate import group_tags

        assert group_tags({"qtype": "keyword", "gold_parts": 1, "gold_chunks": 1}) == ["keyword"]
        assert group_tags({"qtype": "list_all", "gold_parts": 2, "gold_chunks": 4}) == [
            "list_all", "multi_part", "gold4plus"]

    def test_typo_effect_pairs_twins(self):
        from benchmark.v3 import evaluate as E

        arms = {a: {"rank": 1} for a in E.ARMS}
        per_query = {"msp-09": {"arms": {a: {"rank": None} for a in E.ARMS}},
                     "dfa-04": {"arms": arms}}
        out = E.typo_effect(per_query)
        assert list(out) == ["msp-09|dfa-04"]
        assert out["msp-09|dfa-04"]["production"] == {"typo_rank": None, "correct_rank": 1}


# ── freeze ───────────────────────────────────────────────────────────────────

def _queries(n_per_stratum: int):
    out = []
    for dom in ("code", "docs"):
        for qt in ("keyword", "paraphrase"):
            out += [{"id": f"{dom}-{qt}-{i}", "domain": dom, "qtype": qt}
                    for i in range(n_per_stratum)]
    return out


class TestFreeze:
    def test_split_is_stratified_and_deterministic(self):
        from benchmark.v3.freeze import stratified_split

        qs = _queries(10)
        a = stratified_split(qs, seed=7)
        assert a == stratified_split(list(reversed(qs)), seed=7)    # order-independent
        for dom in ("code", "docs"):
            for qt in ("keyword", "paraphrase"):
                ids = [q["id"] for q in qs if q["domain"] == dom and q["qtype"] == qt]
                assert sum(a[i] == "holdout" for i in ids) == 3
        assert set(a) == {q["id"] for q in qs}
        assert stratified_split(qs, seed=8) != a

    def test_verify_detects_a_changed_file(self, tmp_path):
        from benchmark.v3 import freeze

        (tmp_path / "f.txt").write_text("frozen")
        files = {"f.txt": freeze.sha256_file(str(tmp_path / "f.txt"))}
        manifest = {"files": files, "manifest_sha256": freeze._manifest_digest(files)}
        (tmp_path / "m.json").write_text(json.dumps(manifest))
        freeze.verify(str(tmp_path / "m.json"), str(tmp_path))
        (tmp_path / "f.txt").write_text("edited")
        with pytest.raises(freeze.FrozenFileChanged, match="f.txt"):
            freeze.verify(str(tmp_path / "m.json"), str(tmp_path))

    def test_verify_detects_an_edited_file_list(self, tmp_path):
        from benchmark.v3 import freeze

        (tmp_path / "f.txt").write_text("frozen")
        manifest = {"files": {"f.txt": freeze.sha256_file(str(tmp_path / "f.txt"))},
                    "manifest_sha256": "0" * 64}
        (tmp_path / "m.json").write_text(json.dumps(manifest))
        with pytest.raises(freeze.FrozenFileChanged, match="file list"):
            freeze.verify(str(tmp_path / "m.json"), str(tmp_path))


def test_committed_frozen_files_are_intact():
    """CI fails if the frozen query set, split or gold rules are edited without a re-freeze."""
    from benchmark.v3 import freeze

    if not os.path.exists(freeze.MANIFEST_PATH):
        pytest.skip("query set not frozen yet")
    manifest = freeze.verify()
    assignment = freeze.load_split_assignments()
    assert set(assignment) == {q["id"] for q in D.load()["queries"]}
    assert manifest["n_dev"] + manifest["n_holdout"] == len(assignment)
