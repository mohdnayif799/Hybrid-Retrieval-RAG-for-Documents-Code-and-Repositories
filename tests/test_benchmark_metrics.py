"""
Tests for the benchmark's metric arithmetic.

Synthetic fixtures only. These test the SCORING CODE, never the benchmark
results themselves - the results move whenever the corpus or the pin moves,
so asserting on them would produce a test that fails for the wrong reason.

The statistical functions are checked two ways: against values worked out by
hand, and against scipy where an independent implementation exists. Agreeing
with scipy is worth more than agreeing with my own arithmetic.
"""

import math

import pytest
from langchain_core.documents import Document

from benchmark.metrics import (
    discordant_pairs, first_relevant_rank, hit_at_k, locator_of, mcnemar_exact,
    mrr_at_k, normalise_locator, proportion, recall_at_k, wilson_ci,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def repo_chunk(path: str, start_index: int = 0) -> Document:
    return Document(page_content="x", metadata={
        "source": path, "source_type": "repo", "page": 0,
        "start_index": start_index, "start_line": 1})


def upload_chunk(path: str, page: int = 0) -> Document:
    return Document(page_content="x", metadata={
        "source": path, "source_type": "upload", "page": page,
        "start_index": 0})


# ── Locator identity ─────────────────────────────────────────────────────────

class TestLocatorIdentity:
    def test_repo_chunk_keeps_its_full_relative_path(self):
        # Arrange / Act
        loc = locator_of(repo_chunk("src/requests/utils.py"))
        # Assert
        assert loc == ("src/requests/utils.py", None)

    def test_repo_paths_that_share_a_basename_stay_distinct(self):
        a = locator_of(repo_chunk("src/requests/__init__.py"))
        b = locator_of(repo_chunk("tests/__init__.py"))
        assert a != b

    def test_upload_chunk_is_reduced_to_its_basename(self):
        loc = locator_of(upload_chunk(r"C:\some\dir\notes.pdf", page=3))
        assert loc == ("notes.pdf", 3)

    def test_upload_pages_are_distinct_locators(self):
        assert locator_of(upload_chunk("a.pdf", 0)) != locator_of(upload_chunk("a.pdf", 1))

    def test_string_locator_normalises_to_repo_shape(self):
        assert normalise_locator("src/requests/api.py") == ("src/requests/api.py", None)

    def test_mapping_locator_normalises_to_file_page(self):
        assert normalise_locator({"file": "notes.pdf", "page": 2}) == ("notes.pdf", 2)

    def test_yaml_locator_matches_what_a_chunk_reports(self):
        # The whole point: ground truth and retrieval must agree on identity.
        chunk = upload_chunk(r"C:\corpus\notes.pdf", page=1)
        assert locator_of(chunk) == normalise_locator({"file": "notes.pdf", "page": 1})

    def test_unrecognised_locator_raises(self):
        with pytest.raises(ValueError):
            normalise_locator(42)


# ── Rank metrics ─────────────────────────────────────────────────────────────

class TestRankMetrics:
    def test_first_relevant_rank_is_one_based(self):
        assert first_relevant_rank(["a", "b", "c"], {"b"}) == 2

    def test_first_relevant_rank_is_none_when_absent(self):
        assert first_relevant_rank(["a", "b"], {"z"}) is None

    def test_hit_at_k_respects_the_cutoff(self):
        locs = ["a", "b", "c", "d", "e"]
        assert hit_at_k(locs, {"e"}, 5) is True
        assert hit_at_k(locs, {"e"}, 3) is False

    def test_mrr_is_the_reciprocal_of_the_first_hit(self):
        assert mrr_at_k(["a", "b", "c"], {"c"}, 5) == pytest.approx(1 / 3)

    def test_mrr_is_zero_when_nothing_relevant_is_retrieved(self):
        assert mrr_at_k(["a", "b"], {"z"}, 5) == 0.0

    def test_mrr_ignores_hits_beyond_k(self):
        assert mrr_at_k(["a", "b", "c", "d", "e", "f"], {"f"}, 5) == 0.0

    def test_recall_counts_distinct_relevant_locators_found(self):
        # Two of three required locators present in the top 5.
        assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, 5) == pytest.approx(2 / 3)

    def test_recall_does_not_double_count_a_repeated_locator(self):
        assert recall_at_k(["a", "a", "a"], {"a", "b"}, 5) == pytest.approx(0.5)


class TestUndefinedOnEmptyGroundTruth:
    """Hit@K on a query with no correct answer is undefined, NOT zero.

    Returning 0 would silently drag every aggregate down by the number of
    unanswerable queries, which is exactly the bug these guards prevent.
    """

    @pytest.mark.parametrize("fn", [hit_at_k, mrr_at_k, recall_at_k])
    def test_empty_relevant_set_raises_rather_than_scoring_zero(self, fn):
        with pytest.raises(ValueError):
            fn(["a", "b"], set(), 5)


# ── Wilson interval ──────────────────────────────────────────────────────────

class TestWilsonInterval:
    def test_matches_a_hand_worked_value(self):
        # 8/10 at 95%: widely published Wilson bounds are 0.490 and 0.943.
        low, high = wilson_ci(8, 10)
        assert low == pytest.approx(0.490, abs=1e-3)
        assert high == pytest.approx(0.943, abs=1e-3)

    def test_interval_contains_the_point_estimate(self):
        for successes, n in ((0, 7), (1, 7), (4, 39), (39, 39), (5, 5)):
            low, high = wilson_ci(successes, n)
            assert low <= successes / n <= high

    def test_stays_inside_zero_one_at_the_extremes(self):
        # The normal approximation fails here; Wilson must not.
        assert wilson_ci(0, 5)[0] == 0.0
        assert wilson_ci(5, 5)[1] == 1.0
        assert 0.0 <= wilson_ci(0, 5)[1] <= 1.0

    def test_narrows_as_the_sample_grows(self):
        small = wilson_ci(5, 10)
        large = wilson_ci(50, 100)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_zero_sample_returns_a_degenerate_interval(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_successes_above_n_raises(self):
        with pytest.raises(ValueError):
            wilson_ci(6, 5)


# ── McNemar ──────────────────────────────────────────────────────────────────

class TestMcNemarExact:
    def test_no_discordant_pairs_is_no_evidence(self):
        assert mcnemar_exact(0, 0) == 1.0

    def test_symmetric_discordance_is_not_significant(self):
        assert mcnemar_exact(3, 3) == 1.0

    def test_matches_scipy_binomtest_across_a_grid(self):
        # An independent implementation is a stronger check than my own maths.
        scipy_stats = pytest.importorskip("scipy.stats")
        for b in range(0, 9):
            for c in range(0, 9):
                if b + c == 0:
                    continue
                expected = scipy_stats.binomtest(
                    min(b, c), b + c, 0.5, alternative="two-sided").pvalue
                assert mcnemar_exact(b, c) == pytest.approx(min(1.0, expected), abs=1e-9)

    def test_extreme_discordance_is_significant(self):
        # 8 vs 0 discordant pairs: 2 * 0.5**8 = 0.0078
        assert mcnemar_exact(8, 0) == pytest.approx(2 * 0.5 ** 8)

    def test_is_symmetric_in_its_arguments(self):
        assert mcnemar_exact(2, 7) == mcnemar_exact(7, 2)

    def test_small_discordance_cannot_reach_significance(self):
        # The underpowered regime this benchmark actually lands in.
        for b, c in ((2, 1), (3, 0), (1, 2), (0, 4)):
            assert mcnemar_exact(b, c) > 0.05

    def test_negative_counts_raise(self):
        with pytest.raises(ValueError):
            mcnemar_exact(-1, 2)


class TestDiscordantPairs:
    def test_counts_each_direction_separately(self):
        a = [True, True, False, False, True]
        b = [True, False, True, False, False]
        # index1: a won; index2: b won; index4: a won
        assert discordant_pairs(a, b) == (2, 1)

    def test_identical_arms_have_no_discordance(self):
        a = [True, False, True]
        assert discordant_pairs(a, a) == (0, 0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            discordant_pairs([True], [True, False])


# ── Reporting helper ─────────────────────────────────────────────────────────

class TestProportion:
    def test_carries_the_interval_with_the_point_estimate(self):
        p = proportion(34, 39)
        assert p["successes"] == 34 and p["n"] == 39
        assert p["value"] == pytest.approx(34 / 39)
        assert p["ci_low"] < p["value"] < p["ci_high"]

    def test_zero_sample_yields_a_none_value_rather_than_dividing(self):
        assert proportion(0, 0)["value"] is None
