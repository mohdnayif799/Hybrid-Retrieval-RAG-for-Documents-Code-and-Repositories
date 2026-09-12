"""
Tests for range-based ground-truth specifications (v2).

Synthetic fixtures only. Never asserts on the benchmark's measured numbers.

Why range specs exist: Docx2txtLoader returns ONE Document at page 0, so all
99 chunks of the ETCE document share a single (file, page) locator. File-level
ground truth there would be satisfied by any chunk of a 78,000-character
document. These tests pin the matching rule that replaces it.
"""

import pytest
from langchain_core.documents import Document

from benchmark.metrics import (
    chunk_matches, first_relevant_chunk_rank, hit_at_k_chunks, mrr_at_k_chunks,
    normalise_spec, recall_at_k_chunks,
)


def docx_chunk(start: int, length: int = 100, name: str = "unit.docx") -> Document:
    return Document(page_content="x" * length, metadata={
        "source": rf"C:\corpus\{name}", "source_type": "upload",
        "page": 0, "start_index": start})


def repo_chunk(path: str, start: int = 0) -> Document:
    return Document(page_content="y" * 50, metadata={
        "source": path, "source_type": "repo", "page": 0,
        "start_index": start, "start_line": 1})


class TestNormaliseSpec:
    def test_bare_string_is_an_exact_repo_spec(self):
        assert normalise_spec("src/requests/api.py") == ("exact", "src/requests/api.py", None)

    def test_file_page_mapping_is_an_exact_spec(self):
        assert normalise_spec({"file": "a.pdf", "page": 2}) == ("exact", "a.pdf", 2)

    def test_start_end_mapping_is_a_range_spec(self):
        assert normalise_spec({"file": "u.docx", "start": 10, "end": 20}) == \
            ("range", "u.docx", 10, 20)

    def test_range_without_end_collapses_to_a_point(self):
        assert normalise_spec({"file": "u.docx", "start": 7}) == ("range", "u.docx", 7, 7)

    def test_inverted_range_raises(self):
        with pytest.raises(ValueError):
            normalise_spec({"file": "u.docx", "start": 50, "end": 10})

    def test_unrecognised_entry_raises(self):
        with pytest.raises(ValueError):
            normalise_spec(3.14)


class TestChunkMatchesRange:
    SPEC = ("range", "unit.docx", 1000, 2000)

    def test_chunk_fully_inside_the_span_matches(self):
        assert chunk_matches(self.SPEC, docx_chunk(1200, 100)) is True

    def test_chunk_overlapping_the_start_matches(self):
        # Straddles the boundary: carries part of the annotated span.
        assert chunk_matches(self.SPEC, docx_chunk(950, 100)) is True

    def test_chunk_overlapping_the_end_matches(self):
        assert chunk_matches(self.SPEC, docx_chunk(1950, 100)) is True

    def test_chunk_entirely_before_does_not_match(self):
        assert chunk_matches(self.SPEC, docx_chunk(800, 100)) is False

    def test_chunk_entirely_after_does_not_match(self):
        assert chunk_matches(self.SPEC, docx_chunk(2000, 100)) is False

    def test_boundary_is_half_open_at_the_end(self):
        # A chunk starting exactly at `end` carries none of the span.
        assert chunk_matches(self.SPEC, docx_chunk(2000, 50)) is False

    def test_chunk_ending_exactly_at_start_does_not_match(self):
        assert chunk_matches(self.SPEC, docx_chunk(900, 100)) is False

    def test_a_chunk_larger_than_the_span_still_matches(self):
        # The ASCII table spans three chunks; requiring containment would make
        # a span smaller than one chunk unhittable.
        assert chunk_matches(("range", "unit.docx", 1200, 1250),
                             docx_chunk(1000, 500)) is True

    def test_wrong_file_never_matches(self):
        assert chunk_matches(self.SPEC, docx_chunk(1200, 100, name="other.docx")) is False

    def test_chunk_without_start_index_does_not_match(self):
        bad = Document(page_content="x", metadata={"source": r"C:\c\unit.docx",
                                                   "source_type": "upload", "page": 0})
        assert chunk_matches(self.SPEC, bad) is False

    def test_unknown_spec_kind_raises(self):
        with pytest.raises(ValueError):
            chunk_matches(("nonsense", "a", 1, 2), docx_chunk(0))


class TestChunkMatchesExact:
    def test_repo_spec_matches_on_full_path(self):
        assert chunk_matches(("exact", "src/requests/utils.py", None),
                             repo_chunk("src/requests/utils.py")) is True

    def test_repo_spec_rejects_a_same_basename_different_path(self):
        assert chunk_matches(("exact", "src/requests/utils.py", None),
                             repo_chunk("tests/utils.py")) is False


class TestChunkRankMetrics:
    SPECS = [("range", "unit.docx", 1000, 2000)]

    def test_rank_is_one_based_over_chunks(self):
        chunks = [docx_chunk(0), docx_chunk(500), docx_chunk(1500)]
        assert first_relevant_chunk_rank(chunks, self.SPECS) == 3

    def test_hit_respects_the_cutoff(self):
        chunks = [docx_chunk(0), docx_chunk(500), docx_chunk(1500)]
        assert hit_at_k_chunks(chunks, self.SPECS, 3) is True
        assert hit_at_k_chunks(chunks, self.SPECS, 2) is False

    def test_mrr_uses_the_first_matching_chunk(self):
        chunks = [docx_chunk(0), docx_chunk(1500)]
        assert mrr_at_k_chunks(chunks, self.SPECS, 5) == pytest.approx(0.5)

    def test_recall_counts_distinct_specs_satisfied(self):
        specs = [("range", "unit.docx", 0, 100),
                 ("range", "unit.docx", 5000, 5100),
                 ("range", "unit.docx", 9000, 9100)]
        chunks = [docx_chunk(0, 50), docx_chunk(5000, 50)]
        assert recall_at_k_chunks(chunks, specs, 5) == pytest.approx(2 / 3)

    def test_one_chunk_satisfying_two_specs_counts_both(self):
        # The duplicated ETCE block: both copies listed; a chunk in either is
        # a hit, and recall should not punish a query for that.
        specs = [("range", "unit.docx", 0, 200), ("range", "unit.docx", 100, 300)]
        assert recall_at_k_chunks([docx_chunk(150, 10)], specs, 5) == 1.0

    @pytest.mark.parametrize("fn", [hit_at_k_chunks, mrr_at_k_chunks,
                                    recall_at_k_chunks])
    def test_empty_spec_set_raises_rather_than_scoring_zero(self, fn):
        with pytest.raises(ValueError):
            fn([docx_chunk(0)], [], 5)


class TestMixedSpecKinds:
    def test_a_query_can_mix_exact_and_range_specs(self):
        specs = [("exact", "src/requests/api.py", None),
                 ("range", "unit.docx", 1000, 2000)]
        assert first_relevant_chunk_rank(
            [docx_chunk(0), repo_chunk("src/requests/api.py")], specs) == 2
