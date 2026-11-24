import os
import sys

# Repo root on sys.path so `import src.*` works however pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# No test downloads a model. The real cross-encoder reranker and the startup
# model warm-up are switched off for the whole suite; tests that exercise
# reranking inject a fake one (src.reranker.set_reranker_for_tests or the
# HybridRetriever `reranker` field). The built Docker image is where the real
# reranker is loaded and timed (CI "Check the image").
os.environ.setdefault("RAG_RERANKER", "off")
os.environ.setdefault("RAG_WARMUP", "0")
