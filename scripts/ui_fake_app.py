"""
app.py with a fake Gemini client and a fake RAG chain, for browser checks.

    streamlit run scripts/ui_fake_app.py

Nothing reaches the network or a real model. Each question is answered with
one chunk from a file named after the question, so every answer has its own,
distinguishable sources. Used by scripts/ui_sources_check.js; not part of the
test suite and never deployed.
"""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("GOOGLE_API_KEY", "fake-key-for-browser-checks")
os.environ.setdefault("RAG_WARMUP", "0")
os.environ.setdefault("RAG_RERANKER", "off")

from google import genai  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from src import rag_chain  # noqa: E402


class _Response:
    text = "Fake chat title"


class _FakeClient:
    def __init__(self, api_key=None):
        self.models = self

    def generate_content(self, model, contents):
        return _Response()


class _FakeChain:
    def invoke(self, inputs):
        word = inputs["input"].strip(" ?").split()[-1].lower()
        name = f"{word}.txt"
        doc = Document(page_content=f"Everything the documents say about {word}.",
                       metadata={"source": name, "source_type": "upload", "start_index": 0})
        return {"answer": f"An answer about **{word}**, citing {name}.", "context": [doc]}


genai.Client = _FakeClient
rag_chain.build_rag_chain = lambda api_key, model, chroma_dir: _FakeChain()

runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
