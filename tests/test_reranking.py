"""
Two-stage retrieval: the candidate pool, the cross-encoder reranker, and every
fallback that keeps retrieval working when a component is missing or slow.

No model is downloaded: the reranker is a fake with the real Reranker's
interface, and the small-store tests use a real Chroma store with a
deterministic fake embedder.
"""

from __future__ import annotations

import io
import threading

import pytest
from langchain_core.documents import Document

from src import hybrid_retrieval as hr
from src import reranker as rr
from src.hybrid_retrieval import BM25Index, HybridRetriever, minmax_fusion


def make_doc(text: str, start: int, source: str = "notes.txt") -> Document:
    return Document(page_content=text, metadata={"source": source, "page": 0, "start_index": start})


CORPUS = [
    make_doc("The refund policy is described in section four.", 0),
    make_doc("Customers may cancel their subscription at any time.", 100),
    make_doc("Photosynthesis converts light energy into chemical energy.", 200),
    make_doc("Error code CVE-2024-3094 affects the xz compression library.", 300),
    make_doc("Termination of the agreement requires thirty days notice.", 400),
]


class StubVectorRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return list(self.docs)


class FakeModel:
    """Scores a pair by a keyword: a chunk containing it scores high."""

    def __init__(self, keyword="", fail=False):
        self.keyword, self.fail, self.calls = keyword, fail, []

    def predict(self, pairs, batch_size=4):
        self.calls.append(len(pairs))
        if self.fail:
            raise RuntimeError("model exploded")
        return [10.0 if self.keyword and self.keyword in text else -1.0 * i
                for i, (_, text) in enumerate(pairs)]


def retriever(reranker=None, docs=CORPUS, dense=None, bm25=True, **kw):
    return HybridRetriever(
        vector_retriever=StubVectorRetriever(dense if dense is not None else docs),
        bm25_index=BM25Index(docs) if bm25 else None,
        reranker=reranker, log_stream=io.StringIO(), **kw)


# ── the reranker's scoring loop ──────────────────────────────────────────────

class TestRerankerScore:
    def test_scores_in_batches(self):
        model = FakeModel()
        scores = rr.Reranker(model, batch_size=4).score("q", [f"t{i}" for i in range(10)])
        assert len(scores) == 10 and model.calls == [4, 4, 2]

    def test_budget_exceeded_mid_way_raises_timeout(self):
        ticks = iter([0.0, 1.0, 5.0, 9.0, 13.0])          # start, after batch 1, after batch 2, ...
        reranker = rr.Reranker(FakeModel(), batch_size=2)
        with pytest.raises(rr.RerankFailed) as exc:
            reranker.score("q", ["a", "b", "c", "d", "e", "f"], budget_s=3.0, clock=lambda: next(ticks))
        assert exc.value.reason == "timeout"

    def test_budget_exceeded_on_the_last_batch_still_times_out(self):
        ticks = iter([0.0, 4.0, 4.0])
        with pytest.raises(rr.RerankFailed, match="budget"):
            rr.Reranker(FakeModel(), batch_size=4).score("q", ["a", "b"], budget_s=3.0,
                                                          clock=lambda: next(ticks))

    def test_model_error_is_reported_as_error(self):
        with pytest.raises(rr.RerankFailed) as exc:
            rr.Reranker(FakeModel(fail=True)).score("q", ["a"])
        assert exc.value.reason == "error"


class TestLoading:
    def setup_method(self):
        rr.reset_for_tests()

    def teardown_method(self):
        rr.reset_for_tests()

    def test_disabled_by_environment(self, monkeypatch):
        monkeypatch.setenv("RAG_RERANKER", "off")
        assert rr.get_reranker() is None

    def test_load_failure_degrades_to_none(self, monkeypatch):
        sentence_transformers = pytest.importorskip("sentence_transformers")

        def broken(*args, **kwargs):
            raise OSError("model files missing")

        monkeypatch.setenv("RAG_RERANKER", "on")
        monkeypatch.setattr(sentence_transformers, "CrossEncoder", broken)
        assert rr.get_reranker() is None
        assert rr.get_reranker() is None            # cached: no second load attempt

    def test_concurrent_first_calls_load_once(self, monkeypatch):
        sentence_transformers = pytest.importorskip("sentence_transformers")
        loads = []

        def fake_cross_encoder(*args, **kwargs):
            loads.append(1)
            return FakeModel()

        monkeypatch.setenv("RAG_RERANKER", "on")
        monkeypatch.setattr(sentence_transformers, "CrossEncoder", fake_cross_encoder)
        threads = [threading.Thread(target=rr.get_reranker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(loads) == 1 and rr.get_reranker() is not None


# ── min-max fusion ───────────────────────────────────────────────────────────

class TestMinmaxFusion:
    def test_blends_and_records_provenance(self):
        a, b, c, d = CORPUS[:4]
        # dense: a=1.0, b=0.5, d=0.0 after min-max; bm25: b=1.0, c=0.0
        fused = minmax_fusion([(a, 0.9), (b, 0.7), (d, 0.5)], [(b, 12.0), (c, 3.0)], alpha=0.5)
        assert [x.page_content for x in fused] == [
            b.page_content, a.page_content, d.page_content, c.page_content]   # d before c: tie, first seen
        assert [x.metadata["retrieval"] for x in fused] == ["vector+bm25", "vector", "vector", "bm25"]
        assert [x.metadata["fusion_score"] for x in fused] == [0.75, 0.5, 0.0, 0.0]

    def test_single_item_list_scores_one(self):
        fused = minmax_fusion([(CORPUS[0], -3.2)], [], alpha=0.5)
        assert fused[0].metadata["fusion_score"] == 0.5

    def test_ties_keep_first_seen_order_and_inputs_are_not_mutated(self):
        a, b = CORPUS[:2]
        fused = minmax_fusion([(a, 1.0), (b, 1.0)], [], alpha=0.5)
        assert [d.page_content for d in fused] == [a.page_content, b.page_content]
        assert "fusion_score" not in a.metadata

    def test_empty(self):
        assert minmax_fusion([], []) == []


# ── the two-stage retriever ──────────────────────────────────────────────────

class TestTwoStageRetriever:
    def test_reranker_reorders_the_pool(self):
        r = retriever(rr.Reranker(FakeModel("Photosynthesis")))
        results = r.invoke("cancel subscription")
        assert results[0].page_content.startswith("Photosynthesis")
        assert "rerank_score" in results[0].metadata and len(results) <= r.final_k

    def test_only_the_top_rerank_k_are_scored(self):
        model = FakeModel()
        docs = [make_doc(f"chunk number {i} about refunds", i) for i in range(40)]
        r = retriever(rr.Reranker(model), docs=docs, rerank_k=20)
        r.invoke("refunds")
        assert sum(model.calls) == 20

    def test_no_reranker_keeps_fused_order(self):
        r = retriever(None)
        assert [d.page_content for d in r.invoke("refund policy")] == [
            d.page_content for d in r.candidate_pool("refund policy")[:5]]
        assert "fallback=no_reranker" in r.log_stream.getvalue()

    def test_reranker_error_falls_back_to_fused_order(self):
        r = retriever(rr.Reranker(FakeModel(fail=True)))
        assert [d.page_content for d in r.invoke("refund policy")] == [
            d.page_content for d in r.candidate_pool("refund policy")[:5]]
        assert "fallback=error" in r.log_stream.getvalue()

    def test_reranker_over_budget_falls_back_to_fused_order(self):
        class SlowReranker:
            def score(self, query, texts, budget_s):
                raise rr.RerankFailed("timeout")

        r = retriever(SlowReranker())
        assert [d.page_content for d in r.invoke("refund policy")] == [
            d.page_content for d in r.candidate_pool("refund policy")[:5]]
        assert "fallback=timeout" in r.log_stream.getvalue()

    def test_bm25_unavailable_uses_a_dense_only_pool_and_still_reranks(self):
        r = retriever(rr.Reranker(FakeModel("Termination")), bm25=False)
        results = r.invoke("xz library")
        assert results[0].page_content.startswith("Termination")
        assert all(d.metadata["retrieval"] == "vector" for d in r.candidate_pool("xz library"))
        log = r.log_stream.getvalue()
        assert "bm25=off" in log and "fallback=none" in log

    def test_empty_pool_returns_nothing(self):
        r = retriever(rr.Reranker(FakeModel()), dense=[], docs=CORPUS, bm25=False)
        assert r.invoke("anything") == []
        assert "fallback=empty_pool" in r.log_stream.getvalue()

    def test_readme_pin_is_off_by_default(self):
        assert HybridRetriever().surface_readme is False

    def test_log_line_never_contains_the_question_or_chunk_text(self):
        question = "secret question about photosynthesis xyzzy"
        r = retriever(rr.Reranker(FakeModel("Photosynthesis")))
        r.invoke(question)
        log = r.log_stream.getvalue()
        assert log.startswith("retrieval pool=")
        for fragment in ("xyzzy", "secret", "photosynthesis", "Photosynthesis", "refund"):
            assert fragment not in log
        for field in ("pool=", "reranked=", "rerank_ms=", "top1_score=", "fallback=", "bm25="):
            assert field in log


# ── real Chroma stores: tiny, and with BM25 unavailable ─────────────────────

@pytest.fixture
def fake_embeddings(monkeypatch):
    pytest.importorskip("chromadb")
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from src import vector_store
    monkeypatch.setattr(vector_store, "get_embeddings", lambda: DeterministicFakeEmbedding(size=16))
    return vector_store


@pytest.fixture
def fake_reranker():
    rr.set_reranker_for_tests(rr.Reranker(FakeModel("cancel")))
    yield
    rr.reset_for_tests()


def test_tiny_store_uses_whatever_exists(tmp_path, fake_embeddings, fake_reranker, capsys):
    store_dir = str(tmp_path / "tiny")
    fake_embeddings._embed_in_batches(CORPUS[:3], store_dir)
    r = hr.build_hybrid_retriever(store_dir)
    assert r.corpus_size == 3 and r.reranker is not None
    results = r.invoke("cancel subscription refund")
    assert 1 <= len(results) <= 3
    assert results[0].page_content.startswith("Customers may cancel")
    assert "retrieval pool=3 reranked=3" in capsys.readouterr().err


def test_build_without_bm25_still_returns_a_working_retriever(tmp_path, fake_embeddings,
                                                               fake_reranker, monkeypatch, capsys):
    store_dir = str(tmp_path / "nobm25")
    fake_embeddings._embed_in_batches(CORPUS, store_dir)
    monkeypatch.setattr(hr, "BM25_AVAILABLE", False)
    r = hr.build_hybrid_retriever(store_dir)
    assert isinstance(r, HybridRetriever) and r.bm25_index is None
    results = r.invoke("cancel subscription")
    assert results and len(results) <= 5
    assert "bm25=off" in capsys.readouterr().err


def test_build_when_the_bm25_index_fails_to_build(tmp_path, fake_embeddings, fake_reranker,
                                                   monkeypatch, capsys):
    store_dir = str(tmp_path / "bm25fails")
    fake_embeddings._embed_in_batches(CORPUS, store_dir)

    def broken(*args, **kwargs):
        raise RuntimeError("index build failed")

    monkeypatch.setattr(hr, "load_bm25_index", broken)
    r = hr.build_hybrid_retriever(store_dir)
    assert r.bm25_index is None and r.invoke("cancel subscription")


# ── startup warm-up ──────────────────────────────────────────────────────────

def test_warmup_is_disabled_in_the_test_suite():
    assert rr.start_background_warmup() is False


def test_warmup_runs_once_per_process(monkeypatch):
    calls = []
    done = threading.Event()
    monkeypatch.setenv("RAG_WARMUP", "1")
    monkeypatch.setattr(rr, "_WARMUP_STARTED", threading.Event())

    from src import vector_store

    class FakeEmbeddings:
        def embed_query(self, text):
            calls.append("embed")
            return [0.0]

    def fake_get_reranker():
        calls.append("rerank")
        done.set()

    monkeypatch.setattr(vector_store, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(rr, "get_reranker", fake_get_reranker)
    assert rr.start_background_warmup() is True
    assert rr.start_background_warmup() is False
    assert done.wait(5)
    assert calls == ["embed", "rerank"]
