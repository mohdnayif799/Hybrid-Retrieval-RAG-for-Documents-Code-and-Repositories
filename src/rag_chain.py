import re

from google import genai as google_genai
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from src.hybrid_retrieval import build_hybrid_retriever

# ══════════════════════════════════════════════════════════════════════════
# Casual-message detection
# ══════════════════════════════════════════════════════════════════════════
# A lightweight, deterministic, zero-API-cost heuristic that identifies
# high-confidence casual conversation (greetings, thanks, small talk,
# identity/capability questions). Used purely as an OPTIMIZATION to skip
# the retriever for messages that obviously don't need document grounding —
# it is deliberately conservative (uses re.fullmatch on the entire
# normalized message) so compound messages like "Thanks, and what does
# chapter 3 say?" are correctly NOT classified as casual and still reach
# the retriever. A prompt-level safety net (see _QA_TEMPLATE) also handles
# any casual message that slips past this heuristic.

_CASUAL_PATTERNS = [
    r"h(i|ey|ello)+\s*(there)?",
    r"good\s*(morning|afternoon|evening|night)",
    r"how\s*(are|r)\s*(you|u)(\s*doing)?",
    r"how'?s?\s*it\s*going",
    r"what'?s\s*up",
    r"(thanks?|thank\s*you|thx|ty)(\s*so\s*much|\s*a\s*lot)?",
    r"that'?s?\s*(was\s*)?(helpful|great|awesome|nice|perfect|amazing)",
    r"(nice|great|cool|awesome|perfect|good\s*job|well\s*done)",
    r"(ok(ay)?|alright|got\s*it|sure|sounds\s*good)",
    r"(bye|goodbye|see\s*you(\s*later)?|take\s*care|good\s*night)",
    r"what\s*are\s*you",
    r"who\s*are\s*you",
    r"what\s*can\s*you\s*do",
    r"what\s*do\s*you\s*do",
    r"tell\s*me\s*about\s*yourself",
]

_CASUAL_REGEXES = [re.compile(p, re.IGNORECASE) for p in _CASUAL_PATTERNS]


def _is_casual_message(text: str) -> bool:
    """
    Return True only if the ENTIRE message (after normalizing whitespace and
    stripping trailing punctuation) matches a known casual pattern.

    Using re.fullmatch rather than re.search is the key design choice: it
    means a message must be casual from start to end. A message that starts
    casually but contains real content ("Hello, can you summarize chapter
    2?") will correctly fail every pattern and fall through to full RAG
    retrieval instead of being misrouted as small talk.
    """
    normalized = text.strip().lower()
    normalized = re.sub(r"^[!?.,]+", "", normalized)
    normalized = re.sub(r"[!?.,]+$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return False

    for pattern in _CASUAL_REGEXES:
        if pattern.fullmatch(normalized):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════
# Prompt 1: Conversational response (no retrieval)
# ══════════════════════════════════════════════════════════════════════════
# Used when there are no documents at all for this chat, OR when the
# heuristic above is confident the message is casual conversation.

_CONVERSATIONAL_SYSTEM_PROMPT = """You are a friendly, natural conversational assistant that is part of a Retrieval-Augmented Document QA System.

Your personality:
- Warm, natural, and concise -- respond the way a helpful assistant would in normal conversation.
- Do not be robotic or overly formal for casual messages like greetings or thanks.

Your capabilities (explain only if the user actually asks what you are or what you can do):
- You can answer questions grounded in documents the user uploads (PDF, DOCX, PPTX, TXT, MD), citing exact sources.
- You can also have normal conversation, answer general knowledge questions, and remember context within this chat.

CRITICAL RULES:
1. NEVER claim that information came from an uploaded document unless a document was actually retrieved and provided to you as context. You have NOT been given any document context for this message.
2. {doc_status_line}
3. Use the conversation history below to understand references like "that", "this", "it", or "your previous answer".
4. Keep casual replies brief and natural -- do not over-explain.

Conversation history:
{history}

User: {question}

Respond naturally and concisely:"""


def _conversational_response(generate, question: str,
                              chat_history: list, has_documents: bool) -> dict:
    """
    Generate a natural conversational reply without touching the retriever.

    Still receives the full chat history so coreference resolution ("that",
    "this", "it", "your previous answer") works correctly even for casual
    replies -- e.g. "Glad that was helpful!" after a prior document answer.

    Returns the same {"answer": str, "context": list} shape as the RAG path
    so callers never need to branch on which path produced the result.
    Context is always [] here since no retrieval occurred.
    """
    history_text = _history_to_text(chat_history) if chat_history else "(no previous messages)"

    if has_documents:
        doc_status_line = (
            "The user has document(s) uploaded in this chat for other questions, "
            "but this specific message is casual conversation, not a document question."
        )
    else:
        doc_status_line = (
            "No documents have been uploaded in this chat yet. If the user's message "
            "requires specific information from a document (they mention \"the document\", "
            "\"the PDF\", \"the file\", \"chapter X\", \"unit X\", \"according to...\", or ask "
            "you to summarize or list specific uploaded content), tell them plainly and "
            "politely that they need to upload a relevant document first -- do not guess or "
            "invent document content. For general knowledge questions unrelated to a specific "
            "document, you may answer naturally using your own knowledge -- just never claim "
            "the answer came from an uploaded document."
        )

    prompt = _CONVERSATIONAL_SYSTEM_PROMPT.format(
        doc_status_line=doc_status_line,
        history=history_text,
        question=question,
    )
    response = generate(prompt)
    return {"answer": response.text.strip(), "context": []}


# ══════════════════════════════════════════════════════════════════════════
# Prompt 2: Rewrite ambiguous queries for retrieval
# ══════════════════════════════════════════════════════════════════════════

_CONTEXTUALIZE_TEMPLATE = """You are a search query rewriter for a document retrieval system.

Given the conversation history and the user's latest message, rewrite the message
into a fully explicit, self-contained search query.

Rules:
1. Resolve ALL pronouns: it, its, them, they, this, that, those, these, their, he, she, his, her.
2. Resolve positional references: next, previous, next one, next chapter, next unit,
   the first one, the second one, former, latter.
3. If the message is very short (like "next?", "and?", "more?", "them?"), use the
   conversation history to determine what the user is asking about and write a
   complete, specific search query.
4. Do NOT answer the question. Return ONLY the rewritten query.
5. If the message is already fully self-contained, return it exactly as written.

Conversation history:
{history}

User's message: {question}

Rewritten search query (be specific -- expand abbreviations and resolve all references):"""


# ══════════════════════════════════════════════════════════════════════════
# Prompt 3: Generate grounded answer with conversation context
# ══════════════════════════════════════════════════════════════════════════
# Includes a small safety net (final paragraph) for the rare case where a
# casual message slips past the heuristic and reaches this path anyway --
# e.g. an unusual phrasing not covered by _CASUAL_PATTERNS.
#
# The refusal rule (v4 Phase 2b, docs/DECISIONS.md): with the old rule ("if
# the context does not contain the answer, refuse"), 13 of 113 answerable dev
# questions were refused with the gold passage in front of the model. 10 of
# those were bare terms (an identifier or keyword with no question around it)
# or 2-part questions with one part answerable. A bare term is now read as
# "what is it and what does it do", a partly answerable question gets the
# part that can be answered, and the exact refusal sentence is kept for
# questions the context does not answer at all.

_QA_TEMPLATE = """You are a helpful assistant that answers questions strictly from the provided documents.

Use ONLY the context below to answer the question.
If the question is only a term, name or identifier rather than a full question,
treat it as "What is <term> and what does it do?" and answer that from the context.
If the context answers only part of the question, answer that part and say plainly
which part the documents do not cover.
Only if the context contains nothing that answers any part of the question, say exactly:
"I don't have enough information in the provided documents to answer this."
Do not fabricate any information: every statement must be supported by the context.

Exception: if the question is actually casual conversation (a greeting, thanks,
acknowledgment, or small talk) rather than a genuine request for document
information, ignore the context below and respond naturally and warmly instead.
{history_block}
Context from retrieved documents:
{context}

Question: {question}

Answer:"""


def _history_to_text(chat_history: list) -> str:
    lines = []
    for msg in chat_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content
        if role == "Assistant" and len(content) > 800:
            content = content[:800] + "... [truncated]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_rag_chain(gemini_api_key: str, model_name: str, chroma_dir: str | None,
                    llm=None):
    """
    Build a conversational RAG chain using the google.genai SDK directly.
    Compatible with both AQ... and AIza... API key formats.

    chroma_dir may be None -- this happens when a chat has no documents
    uploaded yet. In that case the retriever is never constructed and every
    message is routed through the conversational path.

    Routing per message:
      1. If there are no documents at all, OR the message is a
         high-confidence casual message (greeting, thanks, identity
         question, etc.) -> respond conversationally, skip retrieval
         entirely. This is both a UX improvement (no "I don't have
         information" replies to "thanks!") and an efficiency win (one
         Gemini call instead of two).
      2. Otherwise -> standard RAG pipeline: rewrite the question using
         chat history, retrieve (dense + BM25 candidate pool, reranked by
         a cross-encoder), generate a grounded answer that also
         receives conversation history for reference resolution.

    ``llm`` is for evaluation and tests only: a callable taking a prompt and
    returning an object with a ``.text`` attribute, used instead of the
    Gemini client for every call this chain makes (benchmark/v3/generation
    passes a cached, cost-tracked client). Production passes nothing.
    """
    client = google_genai.Client(api_key=gemini_api_key) if llm is None else None
    has_documents = chroma_dir is not None

    retriever = None
    if has_documents:
        # Two-stage retrieval: a candidate pool of dense search (meaning) and
        # BM25 (exact terms, identifiers, section numbers) fused by score, then
        # a cross-encoder reranker picks the final chunks. See
        # src/hybrid_retrieval.py and docs/DECISIONS.md. Both indexes and the
        # reranker are cached, so rebuilding the chain per message stays cheap.
        retriever = build_hybrid_retriever(chroma_dir)

    def run_pipeline(input_dict: dict) -> dict:
        question     = input_dict["input"]
        chat_history = input_dict.get("chat_history", [])
        # Optional callable, invoked once before every Gemini call, so the
        # caller can log how many calls a query made even when it fails.
        on_gemini_call = input_dict.get("on_gemini_call") or (lambda: None)

        def generate(prompt: str):
            on_gemini_call()
            if llm is not None:
                return llm(prompt)
            return client.models.generate_content(model=model_name, contents=prompt)

        # ── Route 1: no documents at all, or high-confidence casual message ──
        # Never touches the retriever. Guarantees correctness when
        # chroma_dir is None (retriever would not exist to call), and keeps
        # casual replies fast, natural, and free of irrelevant grounding.
        if not has_documents or _is_casual_message(question):
            return _conversational_response(
                generate, question, chat_history, has_documents
            )

        # ── Route 2: standard RAG pipeline ────────────────────────────────
        if chat_history:
            history_text = _history_to_text(chat_history)
            rewrite_prompt = _CONTEXTUALIZE_TEMPLATE.format(
                history=history_text,
                question=question,
            )
            rw = generate(rewrite_prompt)
            standalone = rw.text.strip()
            if len(standalone) > 500 or "\n" in standalone[:50]:
                standalone = question
        else:
            standalone = question

        source_docs = retriever.invoke(standalone)
        context     = "\n\n".join(doc.page_content for doc in source_docs)

        if chat_history:
            history_text = _history_to_text(chat_history)
            history_block = (
                "\nConversation history (use this to resolve references like "
                "\"them\", \"it\", \"next\", etc. -- do NOT use it as a source of facts):\n"
                + history_text + "\n"
            )
        else:
            history_block = ""

        qa_prompt = _QA_TEMPLATE.format(
            context=context,
            question=question,
            history_block=history_block,
        )
        qa = generate(qa_prompt)
        return {"answer": qa.text.strip(), "context": source_docs}

    return RunnableLambda(run_pipeline)
