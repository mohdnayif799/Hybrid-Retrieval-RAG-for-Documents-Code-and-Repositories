"""
End-to-end integration: archive -> documents -> chunks -> hybrid retrieval -> citations.

Everything except the Chroma/embedding layer is the real code. The dense side
is a stub returning a fixed ranked list, which is what lets this run with no
torch, no Chroma and no API key while still exercising BM25, RRF, metadata
propagation and citation rendering together.
"""

import io
import tarfile
import time

from langchain_core.documents import Document

from src.chunking import chunk_documents
from src.citations import build_inline_citations, format_reference
from src.hybrid_retrieval import BM25Index, HybridRetriever, chunk_key
from src.repo_ingestion import RepoRef, read_archive, repo_files_to_documents

ROOT = "acme-service-deadbeef"

REPO = {
    "src/auth/tokens.py": (
        "import time\n\n\n"
        "def validate_access_token(token, secret):\n"
        "    \"\"\"Check an access token's signature and expiry.\"\"\"\n"
        "    if not token:\n"
        "        return False\n"
        "    return _verify(token, secret)\n\n\n"
        "def refresh_access_token(refresh_token):\n"
        "    return _issue(refresh_token)\n"
    ),
    "src/billing/invoices.ts": (
        "export function calculateInvoiceTotal(items: Item[]): number {\n"
        "  return items.reduce((sum, i) => sum + i.price, 0);\n"
        "}\n\n"
        "export function applyDiscountCode(total: number, code: string): number {\n"
        "  return code === 'SAVE10' ? total * 0.9 : total;\n"
        "}\n"
    ),
    "src/auth/__init__.py": "from .tokens import validate_access_token\n",
    "src/billing/__init__.py": "from .invoices import calculateInvoiceTotal\n",
    "README.md": "# Acme Service\n\nHandles authentication and billing.\n",
    "node_modules/junk/index.js": "module.exports = 1;\n",
    "package-lock.json": '{"lockfileVersion": 3}\n',
}


def build_repo_tarball(entries=None):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, content in (entries or REPO).items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=f"{ROOT}/{path}")
            info.size = len(data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def ingest():
    files, report = read_archive(build_repo_tarball())
    docs = repo_files_to_documents(files, RepoRef("acme", "service"), "main")
    return chunk_documents(docs), report


class StubVectorRetriever:
    def __init__(self, docs):
        self.docs = list(docs)

    def invoke(self, query):
        return self.docs


# ── Pipeline ──────────────────────────────────────────────────────────────────

def test_full_ingestion_produces_usable_chunks():
    chunks, report = ingest()
    assert report.kept == 5                       # junk and lockfile filtered
    assert chunks
    sources = {c.metadata["source"] for c in chunks}
    assert "src/auth/tokens.py" in sources
    assert "node_modules/junk/index.js" not in sources


def test_every_metadata_value_is_chroma_compatible():
    # Chroma only accepts str/int/float/bool/None in metadata. A list or dict
    # here would blow up at real ingestion time, long after these tests pass.
    chunks, _ = ingest()
    for chunk in chunks:
        for key, value in chunk.metadata.items():
            assert isinstance(value, (str, int, float, bool)) or value is None, \
                f"{key}={value!r} ({type(value).__name__}) is not Chroma-compatible"


def test_chunk_keys_are_unique_across_the_repository():
    # RRF dedup depends on this; a collision would silently drop a chunk.
    chunks, _ = ingest()
    keys = [chunk_key(c) for c in chunks]
    assert len(keys) == len(set(keys))


def test_identically_named_init_files_remain_distinct():
    chunks, _ = ingest()
    inits = [c for c in chunks if c.metadata["source"].endswith("__init__.py")]
    assert len({c.metadata["source"] for c in inits}) == 2


# ── Hybrid retrieval over repository content ──────────────────────────────────

def test_bm25_reaches_a_snake_case_function_from_natural_language():
    chunks, _ = ingest()
    hits = BM25Index(chunks).search("validate access token", k=5)
    assert hits
    assert "validate_access_token" in hits[0][0].page_content


def test_bm25_reaches_a_camel_case_function_from_natural_language():
    # This is the case that returned nothing before the tokenizer fix.
    chunks, _ = ingest()
    hits = BM25Index(chunks).search("calculate invoice total", k=5)
    assert hits, "camelCase identifier was unreachable"
    assert "calculateInvoiceTotal" in hits[0][0].page_content


def test_hybrid_promotes_a_lexical_hit_the_dense_side_missed():
    chunks, _ = ingest()
    target = "applyDiscountCode"
    dense_only = [c for c in chunks if target not in c.page_content]

    retriever = HybridRetriever(
        vector_retriever=StubVectorRetriever(dense_only),
        bm25_index=BM25Index(chunks),
        final_k=5,
    )
    results = retriever.invoke("apply discount code")
    assert any(target in r.page_content for r in results), \
        "BM25 failed to rescue an identifier the dense retriever missed"


def test_retrieved_repo_chunks_keep_citation_metadata():
    chunks, _ = ingest()
    retriever = HybridRetriever(
        vector_retriever=StubVectorRetriever(chunks),
        bm25_index=BM25Index(chunks),
        final_k=3,
    )
    for doc in retriever.invoke("validate access token"):
        assert doc.metadata["source_type"] == "repo"
        assert doc.metadata["repo"] == "acme/service"
        assert doc.metadata["ref"] == "main"
        assert isinstance(doc.metadata["start_line"], int)
        assert "retrieval" in doc.metadata


def test_retrieval_output_renders_as_path_and_line_citations():
    chunks, _ = ingest()
    retriever = HybridRetriever(
        vector_retriever=StubVectorRetriever(chunks),
        bm25_index=BM25Index(chunks),
        final_k=3,
    )
    results = retriever.invoke("validate access token")
    rendered = build_inline_citations(results)

    assert "acme/service" in rendered
    assert any("/" in d.metadata["source"] for d in results)
    ref = format_reference(results[0], 1)
    assert ref.count(":") == 1 and ref.split(":")[1].isdigit()


# ── Mixed corpus: documents and repository together ───────────────────────────

def test_documents_and_repository_coexist_in_one_index():
    repo_chunks, _ = ingest()
    doc_chunks = chunk_documents([
        Document(
            page_content="The refund policy is described in Section 4.2. " * 30,
            metadata={"source": "policy.pdf", "page": 2},
        )
    ])
    everything = repo_chunks + doc_chunks
    index = BM25Index(everything)

    code_hits = index.search("validate access token", k=3)
    doc_hits = index.search("refund policy Section 4.2", k=3)

    assert code_hits and code_hits[0][0].metadata["source_type"] == "repo"
    assert doc_hits and doc_hits[0][0].metadata["source_type"] == "upload"


def test_existing_document_ingestion_is_unaffected_by_repo_support():
    doc = Document(page_content="Prose paragraph. " * 200,
                   metadata={"source": "manual.pdf", "page": 0})
    chunks = chunk_documents([doc])
    assert all(c.metadata["source"] == "manual.pdf" for c in chunks)
    assert all("start_line" not in c.metadata for c in chunks)
    assert all(len(c.page_content) <= 1000 for c in chunks)
