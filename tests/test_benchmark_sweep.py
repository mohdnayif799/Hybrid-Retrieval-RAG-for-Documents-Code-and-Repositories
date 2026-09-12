"""
Tests for the configuration-sweep logic.

Synthetic fixtures only. These test the sweep's SCORING RULE, never the
sweep's measured numbers - those are measurements of a particular corpus and
would change the moment the pin moves, so asserting on them would produce a
test that fails for the wrong reason.
"""

import pytest
from langchain_core.documents import Document

from benchmark.sweep import CONFIGS, bm25_median_gate


def hit(name: str, score: float):
    """A (Document, score) pair shaped exactly like BM25Index.search returns."""
    return (Document(page_content=name, metadata={"source": name}), score)


def names(docs):
    return [d.page_content for d in docs]


class TestBm25MedianGate:
    def test_empty_candidate_list_yields_nothing(self):
        assert bm25_median_gate([]) == []

    def test_single_candidate_is_kept(self):
        # Its own score is the median, and the rule keeps >= median.
        assert names(bm25_median_gate([hit("a", 4.2)])) == ["a"]

    def test_odd_count_keeps_upper_half_including_the_median(self):
        # Arrange: scores 5,4,3,2,1 -> median 3
        hits = [hit("a", 5), hit("b", 4), hit("c", 3), hit("d", 2), hit("e", 1)]
        # Act / Assert
        assert names(bm25_median_gate(hits)) == ["a", "b", "c"]

    def test_even_count_keeps_the_upper_half(self):
        # scores 4,3,2,1 -> median 2.5
        hits = [hit("a", 4), hit("b", 3), hit("c", 2), hit("d", 1)]
        assert names(bm25_median_gate(hits)) == ["a", "b"]

    def test_ties_at_the_median_are_all_kept(self):
        # scores 3,3,3,1 -> median 3.0; the rule is >=, so three survive.
        hits = [hit("a", 3), hit("b", 3), hit("c", 3), hit("d", 1)]
        assert names(bm25_median_gate(hits)) == ["a", "b", "c"]

    def test_uniform_scores_keep_every_candidate(self):
        hits = [hit(n, 2.0) for n in "abcd"]
        assert names(bm25_median_gate(hits)) == ["a", "b", "c", "d"]

    def test_negative_scores_are_handled(self):
        # Okapi IDF goes negative on small corpora, which the project
        # documents; the gate must not assume positivity.
        hits = [hit("a", -1.0), hit("b", -2.0), hit("c", -3.0)]
        assert names(bm25_median_gate(hits)) == ["a", "b"]

    def test_input_order_is_preserved(self):
        hits = [hit("a", 1), hit("b", 9), hit("c", 5)]
        # median 5 -> keeps b and c, in their original relative order
        assert names(bm25_median_gate(hits)) == ["b", "c"]

    def test_returns_documents_not_score_pairs(self):
        out = bm25_median_gate([hit("a", 1.0)])
        assert all(isinstance(d, Document) for d in out)

    def test_never_returns_more_than_it_was_given(self):
        hits = [hit(str(i), float(i)) for i in range(10)]
        assert len(bm25_median_gate(hits)) <= len(hits)

    def test_keeps_at_least_one_candidate_when_any_exist(self):
        # A gate that could empty a non-empty list would silently turn hybrid
        # retrieval into dense-only without any signal that it had.
        for n in range(1, 12):
            hits = [hit(str(i), float(i)) for i in range(n)]
            assert len(bm25_median_gate(hits)) >= 1


class TestPreRegisteredConfigs:
    """The configurations are pre-registered; these pin them so an accidental
    edit shows up as a failing test rather than a silently different sweep."""

    def test_all_registered_configs_present(self):
        # v1 pre-registered four; v2 added two reweight configs.
        assert set(CONFIGS) == {"C0_baseline", "C1_k6", "C2_k8", "C3_gated",
                                "W1_55_45", "W2_60_40"}

    @pytest.mark.parametrize("name,final_k,gate", [
        ("C0_baseline", 5, False),
        ("C1_k6", 6, False),
        ("C2_k8", 8, False),
        ("C3_gated", 5, True),
        ("W1_55_45", 5, False),
        ("W2_60_40", 5, False),
    ])
    def test_config_parameters_are_as_registered(self, name, final_k, gate):
        assert CONFIGS[name]["final_k"] == final_k
        assert CONFIGS[name]["gate"] is gate

    @pytest.mark.parametrize("name,wv,wb", [
        ("W1_55_45", 0.55, 0.45),
        ("W2_60_40", 0.60, 0.40),
    ])
    def test_reweight_configs_carry_their_registered_weights(self, name, wv, wb):
        assert CONFIGS[name]["vector_weight"] == wv
        assert CONFIGS[name]["bm25_weight"] == wb

    def test_baseline_configs_do_not_override_shipped_weights(self):
        # C0/C1/C2/C3 must inherit the shipped 0.5/0.5, not restate it.
        for name in ("C0_baseline", "C1_k6", "C2_k8", "C3_gated"):
            assert "vector_weight" not in CONFIGS[name]
            assert "bm25_weight" not in CONFIGS[name]

    def test_every_config_records_its_rationale(self):
        assert all(c["why"].strip() for c in CONFIGS.values())
