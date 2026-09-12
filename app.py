import html
import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from google import genai as google_genai
from langchain_core.messages import HumanMessage, AIMessage

from src.data_ingestion import load_documents, chunk_documents, SUPPORTED_EXTENSIONS
from src.vector_store import (
    create_chat_vector_store,
    append_to_vector_store,
    get_store_stats,
)
from src.rag_chain import build_rag_chain
from src.citations import (
    build_inline_citations,
    doc_type_label,
    format_location,
    format_reference,
    retrieval_badge,
)
from src.repo_ingestion import (
    RepositoryError,
    load_repository,
    parse_repo_url,
)
from demo_key import get_api_key, limit_reached, increment_query_count, queries_remaining

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document QA System",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Active chat item */
.active-chat-item {
    background: linear-gradient(90deg, rgba(79,70,229,0.13) 0%, rgba(79,70,229,0.04) 100%);
    border-left: 3px solid #4F46E5;
    border-radius: 0 8px 8px 0;
    padding: 9px 14px;
    font-weight: 600;
    color: #4F46E5;
    margin: 2px 0 2px 0;
    font-size: 14px;
    line-height: 1.4;
    cursor: default;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
/* Inactive chat buttons */
div[data-testid="stSidebar"] div.stButton button {
    border-radius: 8px;
    font-size: 14px;
    text-align: left;
}
/* Status box */
.status-box {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    line-height: 1.7;
}
.status-box-warn {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    line-height: 1.7;
}
/* ── User message bubble ──────────────────────────────────────────────────
   Right-aligned, no avatar. The row is a flex container and the bubble is a
   flex item, so the bubble is sized by its CONTENT and only grows to the cap —
   a three-word question stays three words wide instead of becoming a
   full-width block.

   max-width uses min() of a percentage and an absolute cap so it is
   responsive in both directions: the rem cap stops the bubble sprawling on a
   wide monitor, the percentage keeps it proportional on a small one. At
   layout="wide" the 36rem cap normally wins, leaving roughly the left 60% of
   the reading column empty. No fixed pixel offsets anywhere. */
.user-msg-row {
    display: flex;
    justify-content: flex-end;
    margin: 1.75rem 0 0.55rem 0;
}
.user-msg {
    max-width: min(56%, 36rem);
    background: rgba(59, 130, 246, 0.10);
    border: 1px solid rgba(59, 130, 246, 0.20);
    border-radius: 1.15rem 1.15rem 0.35rem 1.15rem;
    padding: 0.7rem 1.05rem;
    font-size: 0.95rem;
    line-height: 1.55;
    text-align: left;
    white-space: pre-wrap;      /* keep newlines in multi-line questions */
    overflow-wrap: anywhere;    /* never let a long URL widen the bubble */
}
@media (max-width: 640px) {
    .user-msg { max-width: 86%; }
}
</style>
""", unsafe_allow_html=True)


# ── Citation display ─────────────────────────────────────────────────────────
# Formatting lives in src/citations.py so it can be unit tested — importing
# app.py would execute this whole Streamlit script.


# ── Helper: LLM-generated chat title ─────────────────────────────────────────

def _generate_title(question: str, api_key: str, model: str) -> str:
    """
    Use Gemini to generate a concise 3-5 word title from the first question.
    Falls back to truncated question if the API call fails.
    """
    try:
        client = google_genai.Client(api_key=api_key)
        prompt = (
            "Generate a concise chat title (3 to 5 words maximum) that captures "
            "the topic of this question. Return ONLY the title — no punctuation, "
            "no quotes, no explanation.\n\n"
            f"Question: {question}"
        )
        resp  = client.models.generate_content(model=model, contents=prompt)
        title = resp.text.strip().strip('"\'').strip()
        words = title.split()
        if len(words) > 6:
            title = " ".join(words[:5]) + "..."
        return title if title else _auto_title(question)
    except Exception:
        return _auto_title(question)


# ── State helpers ─────────────────────────────────────────────────────────────

def _blank_chat() -> dict:
    return {
        "title":          "New Chat",
        "messages":       [],
        "lc_history":     [],
        "chroma_dir":     None,
        "uploaded_files": [],
        "repos":          [],
        "response_cache": {},
    }


def _init():
    if "chats" not in st.session_state:
        first_id = str(uuid.uuid4())
        st.session_state.chats      = {first_id: _blank_chat()}
        st.session_state.chat_order = [first_id]
        st.session_state.active_id  = first_id


def _active() -> dict:
    return st.session_state.chats[st.session_state.active_id]


def _auto_title(text: str) -> str:
    text = text.strip()
    return (text[:30] + "...") if len(text) > 30 else text


def _render_user_message(text: str) -> None:
    """
    Draw a user turn as a right-aligned bubble with no avatar.

    Text is HTML-escaped rather than passed through st.markdown. Beyond the
    obvious injection reason, it means a question containing __init__.py or
    *asterisks* is shown exactly as typed instead of being reinterpreted as
    formatting — the same class of problem that mangled repository citations.
    """
    st.markdown(
        f'<div class="user-msg-row"><div class="user-msg">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_sources(source_docs) -> None:
    """
    Draw the Sources block and the source-chunk expander for one answer.

    Single renderer shared by the history replay loop and the live response.
    Previously this markup existed only in the live path, so sources vanished
    from every earlier answer on the next rerun — the duplication was the bug,
    so there is now exactly one copy.
    """
    if not source_docs:
        return

    st.markdown("---\n**Sources:**\n\n" + build_inline_citations(source_docs))

    with st.expander("\U0001f4da Source Chunks Used"):
        for i, doc in enumerate(source_docs, 1):
            sf    = doc.metadata.get("source", "unknown")
            stype = doc.metadata.get("source_type")
            st.markdown(
                f"**Chunk {i}** \u2014 `{format_reference(doc, i)}` "
                f"· {doc_type_label(sf, stype)} · **{format_location(doc, i)}**"
                f"{retrieval_badge(doc)}"
            )
            st.caption(doc.page_content[:500] + "\u2026")
            st.divider()


def _index_chunks(aid: str, chunks: list, progress_bar, base_pct: int = 60):
    """
    Embed and store chunks, creating the chat's store on first ingestion and
    appending afterwards. Shared by the document and repository paths so both
    get identical store lifecycle handling and embedding progress.
    """
    chat = st.session_state.chats[aid]
    total = max(len(chunks), 1)

    def embed_progress(done, _total):
        pct = base_pct + int((done / total) * (99 - base_pct))
        progress_bar.progress(
            min(pct, 99), text=f"Embedding {min(done, total)}/{total} chunks\u2026"
        )

    if chat["chroma_dir"] is None:
        _, chroma_dir = create_chat_vector_store(chunks, progress_callback=embed_progress)
        chat["chroma_dir"] = chroma_dir
    else:
        append_to_vector_store(chunks, chat["chroma_dir"], progress_callback=embed_progress)


def _new_chat():
    nid = str(uuid.uuid4())
    st.session_state.chats[nid] = _blank_chat()
    st.session_state.chat_order.insert(0, nid)
    st.session_state.active_id = nid


# ── Bootstrap ─────────────────────────────────────────────────────────────────
_init()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # Configuration
    st.markdown("### \u2699\ufe0f Configuration")

    user_gemini_key = st.text_input(
        "Your own Gemini API key (optional)",
        type="password",
        help="Leave blank to use the shared demo key. Get a free key at "
             "https://aistudio.google.com/apikey",
    )

    gemini_api_key = get_api_key(user_gemini_key)

    if limit_reached(user_gemini_key):
        st.error(
            "Shared demo key limit reached (5 queries this session). "
            "Enter your own key above to keep going."
        )
    elif not user_gemini_key and gemini_api_key:
        st.caption(
            f"Using the shared demo key \u2014 "
            f"{queries_remaining(user_gemini_key)} of 5 queries left this session."
        )
    model_name = st.selectbox(
        "Model",
        options=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
        index=0,
    )

    st.divider()

    # New Chat
    if st.button("\u2795  New Chat", width="stretch"):
        _new_chat()
        st.rerun()

    # Chat list with active highlighting
    if st.session_state.chat_order:
        st.markdown("**Chats**")
        for cid in st.session_state.chat_order:
            c      = st.session_state.chats[cid]
            active = cid == st.session_state.active_id
            title  = c["title"]
            if active:
                # Styled div — no click needed, already active
                st.markdown(
                    f'<div class="active-chat-item">\u25b6&nbsp;&nbsp;{title}</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(
                    f"\u00a0\u00a0\u00a0{title}",
                    key=f"sw_{cid}",
                    width="stretch",
                ):
                    st.session_state.active_id = cid
                    st.rerun()

    st.divider()

    # Upload Documents
    # ── Improvement 4 fix: unique key per chat ─────────────────────────────
    # When the active chat changes, Streamlit creates a new uploader widget
    # (blank) for the new chat's ID. This prevents documents from one chat
    # visually carrying over to another chat's uploader.
    aid_for_key = st.session_state.active_id

    st.markdown("**\U0001f4c1 Upload Documents**")
    st.caption(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    uploaded_files = st.file_uploader(
        "Drop files here or click Browse",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{aid_for_key}",    # unique per chat — resets on chat switch
    )

    process_btn = st.button(
        "\u26a1 Process Documents",
        disabled=not uploaded_files,
        width="stretch",
        type="primary",
    )

    if process_btn:
        if not gemini_api_key:
            st.error("Enter your Gemini API key above.")
        else:
            aid         = st.session_state.active_id
            known_files = st.session_state.chats[aid]["uploaded_files"]

            new_uploads  = [f for f in uploaded_files if f.name not in known_files]
            already_have = [f.name for f in uploaded_files if f.name in known_files]

            if already_have:
                st.info(f"Already indexed, skipping: {', '.join(already_have)}")

            if not new_uploads:
                st.warning("No new documents to process.")
            else:
                with st.spinner(f"Processing {len(new_uploads)} file(s)\u2026"):
                    try:
                        temp_paths, original_names = [], []
                        for f in new_uploads:
                            suffix = os.path.splitext(f.name)[1]
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=suffix
                            ) as tmp:
                                tmp.write(f.read())
                                temp_paths.append(tmp.name)
                                original_names.append(f.name)

                        progress_bar = st.progress(0, text="Reading pages\u2026")

                        def update_progress(current, total):
                            pct = int((current / total) * 80)
                            progress_bar.progress(
                                pct, text=f"Reading page {current}/{total}\u2026"
                            )

                        docs = load_documents(temp_paths, progress_callback=update_progress)

                        for doc in docs:
                            for tp, on in zip(temp_paths, original_names):
                                if doc.metadata.get("source") == tp:
                                    doc.metadata["source"] = on
                                    break

                        progress_bar.progress(85, text="Chunking text\u2026")
                        chunks = chunk_documents(docs)

                        # Shared with the repository path: creates the store on
                        # first ingestion, appends afterwards, reports progress.
                        _index_chunks(aid, chunks, progress_bar, base_pct=88)

                        st.session_state.chats[aid]["uploaded_files"].extend(original_names)

                        for p in temp_paths:
                            os.unlink(p)

                        progress_bar.progress(100, text="Done!")
                        n_docs = len(st.session_state.chats[aid]["uploaded_files"])
                        st.success(
                            f"\u2705 Indexed **{len(chunks)} new chunks**.\n"
                            f"This chat now has **{n_docs} document(s)** searchable."
                        )

                    except Exception as e:
                        st.error(f"Failed to process documents: {e}")

    st.divider()

    # ── GitHub repository ingestion ──────────────────────────────────────────
    # Deliberately a separate control from the file uploader: the inputs, the
    # failure modes and the time they take are all different, and merging them
    # would hide a multi-minute network operation behind a "Process" button.
    st.markdown("**\U0001f419 Add GitHub Repository**")
    st.caption(
        "Public repositories. Source and docs are indexed; build output, "
        "lockfiles, binaries and generated files are skipped."
    )

    repo_url = st.text_input(
        "GitHub repository URL",
        placeholder="https://github.com/owner/repo",
        label_visibility="collapsed",
        key=f"repo_url_{aid_for_key}",
    )
    with st.expander("Private repo or rate-limited?"):
        github_token = st.text_input(
            "GitHub personal access token (optional)",
            type="password",
            help="Raises the GitHub API limit from 60 to 5,000 requests/hour. "
                 "Never stored — used only for this request.",
            key=f"gh_token_{aid_for_key}",
        )

    repo_btn = st.button(
        "\U0001f4e5 Ingest Repository",
        disabled=not repo_url,
        width="stretch",
    )

    if repo_btn:
        aid = st.session_state.active_id
        chat = st.session_state.chats[aid]
        chat.setdefault("repos", [])

        if not gemini_api_key:
            st.error("Enter your Gemini API key above.")
        else:
            try:
                # Parse first: catches a bad URL with no network call at all.
                parsed = parse_repo_url(repo_url)
                already = [r for r in chat["repos"] if r.startswith(parsed.slug + "@")]
            except RepositoryError as exc:
                parsed, already = None, []
                st.error(str(exc))

            if parsed is not None:
                if already:
                    st.info(f"Already indexed in this chat: {already[0]}")
                else:
                    progress_bar = st.progress(0, text="Resolving repository\u2026")

                    def on_phase(phase, current, _total):
                        if phase == "resolve":
                            progress_bar.progress(3, text="Resolving repository\u2026")
                        elif phase == "download":
                            mb = current / (1024 * 1024)
                            progress_bar.progress(
                                min(35, 5 + int(mb * 3)),
                                text=f"Downloading archive\u2026 {mb:.1f} MB",
                            )
                        elif phase == "read":
                            progress_bar.progress(
                                45, text=f"Reading files\u2026 {current} kept"
                            )

                    try:
                        docs, repo, ref, report = load_repository(
                            repo_url,
                            token=(github_token or None),
                            on_phase=on_phase,
                        )

                        progress_bar.progress(55, text="Chunking source files\u2026")
                        chunks = chunk_documents(docs)

                        _index_chunks(aid, chunks, progress_bar, base_pct=60)

                        label = repo.label(ref)
                        chat["repos"].append(label)
                        progress_bar.progress(100, text="Done!")

                        skipped = report.skipped_total
                        st.success(
                            f"\u2705 Indexed **{label}**\n\n"
                            f"{report.kept} file(s) \u2192 **{len(chunks)} chunks**"
                            + (f" · {skipped} file(s) filtered out" if skipped else "")
                        )
                        if report.skipped:
                            with st.expander("What was filtered out"):
                                for reason, count in sorted(
                                    report.skipped.items(), key=lambda kv: -kv[1]
                                ):
                                    st.caption(f"\u2022 {count} \u00d7 {reason}")

                    except RepositoryError as exc:
                        # Expected, user-actionable failures: bad URL, private
                        # repo, rate limit, oversized repo, nothing indexable.
                        progress_bar.empty()
                        st.error(str(exc))
                    except Exception as exc:
                        progress_bar.empty()
                        st.error(f"Unexpected error ingesting repository: {exc}")

    st.divider()

    # Active Documents list
    active_chat = _active()
    st.markdown("**\U0001f4cb Active Sources**")
    active_repos = active_chat.get("repos", [])
    if active_chat["uploaded_files"] or active_repos:
        for fname in active_chat["uploaded_files"]:
            st.caption(f"\u2022 {fname}  ·  {doc_type_label(fname)}")
        for label in active_repos:
            st.caption(f"\u2022 {label}  ·  \U0001f419 Repository")
    else:
        st.caption("Nothing indexed yet — you can still chat, or add a document "
                   "or GitHub repository for grounded answers.")

    st.divider()

    # ── Status Indicator ─────────────────────────────────────────────────────
    # "No documents" is now a green ready state, not an amber warning —
    # casual conversation and general questions work without any upload.
    st.markdown("**\U0001f4e1 System Status**")
    if not gemini_api_key:
        status_html = (
            '<div class="status-box-warn">\U0001f7e1 <b>Waiting</b><br>'
            'Enter Gemini API key to begin.</div>'
        )
    elif not active_chat["chroma_dir"]:
        status_html = (
            '<div class="status-box">\U0001f7e2 <b>Ready to Chat</b><br>'
            f'Model: {model_name}<br>'
            'Nothing indexed yet \u2014 add a document or repository for cited answers.</div>'
        )
    else:
        stats   = get_store_stats(active_chat["chroma_dir"])
        total   = stats.get("total_chunks", 0)
        n_docs  = len(active_chat["uploaded_files"])
        n_repos = len(active_chat.get("repos", []))
        n_files = len(stats.get("per_doc", {}))
        source_line = f"Documents: {n_docs} indexed"
        if n_repos:
            source_line += f"<br>Repositories: {n_repos} ({n_files} files)"
        status_html = (
            '<div class="status-box">\U0001f7e2 <b>Ready</b><br>'
            f'{source_line}<br>'
            f'Chunks in store: {total}<br>'
            f'Model: {model_name}</div>'
        )
    st.markdown(status_html, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Hybrid-Retrieval RAG for Documents & Code Repositories")
st.caption("Chat naturally, or upload documents in the sidebar for grounded, cited answers.")

active_chat = _active()

for msg in active_chat["messages"]:
    if msg["role"] == "user":
        _render_user_message(msg["content"])
    else:
        # Plain container, not st.chat_message: chat_message always renders an
        # avatar and there is no supported way to suppress it. Hiding it with
        # CSS would mean targeting Streamlit's internal test IDs, which change
        # between releases. A container gives the same grouping with no chrome.
        with st.container():
            st.markdown(msg["content"])
            # .get() so messages stored before sources were persisted, and
            # error messages (which carry none), still replay cleanly.
            _render_sources(msg.get("sources") or [])

# ── Chat input ────────────────────────────────────────────────────────────────
if user_question := st.chat_input("Ask a question, or just say hello\u2026"):
    if not gemini_api_key:
        st.warning("\u26a0\ufe0f Enter your Gemini API key in the sidebar.")
    else:
        # No blocking check on chroma_dir here — build_rag_chain accepts
        # chroma_dir=None and routes every message through the conversational
        # path when this chat has no documents uploaded yet.
        aid = st.session_state.active_id

        # ── Improvement 2: LLM-generated title on first message ───────────────
        if not st.session_state.chats[aid]["messages"]:
            st.session_state.chats[aid]["title"] = _generate_title(
                user_question, gemini_api_key, model_name
            )

        st.session_state.chats[aid]["messages"].append(
            {"role": "user", "content": user_question}
        )
        _render_user_message(user_question)

        with st.container():          # see replay loop: avoids the avatar
            try:
                cache_key = user_question.strip().lower()
                cache     = st.session_state.chats[aid]["response_cache"]

                if cache_key in cache:
                    answer      = cache[cache_key]["answer"]
                    source_docs = cache[cache_key]["source_docs"]
                    st.caption("\U0001f5c4\ufe0f *(retrieved from session cache)*")
                else:
                    with st.spinner("Searching documents and generating answer\u2026"):
                        chain = build_rag_chain(
                            gemini_api_key,
                            model_name,
                            st.session_state.chats[aid]["chroma_dir"],
                        )
                        result = chain.invoke({
                            "input":        user_question,
                            "chat_history": st.session_state.chats[aid]["lc_history"],
                        })
                        answer      = result["answer"]
                        source_docs = result["context"]
                        cache[cache_key] = {"answer": answer, "source_docs": source_docs}
                        increment_query_count(user_gemini_key)  # only a real Gemini call counts

                st.markdown(answer)

                _render_sources(source_docs)

                # Sources are stored ON the message, not in a single "latest
                # answer" variable, so every turn keeps its own citations for
                # the life of the conversation.
                st.session_state.chats[aid]["messages"].append(
                    {"role": "assistant", "content": answer,
                     "sources": list(source_docs or [])}
                )
                st.session_state.chats[aid]["lc_history"].extend([
                    HumanMessage(content=user_question),
                    AIMessage(content=answer),
                ])

            except Exception as e:
                error_msg = f"Error generating answer: {e}"
                st.error(error_msg)
                st.session_state.chats[aid]["messages"].append(
                    {"role": "assistant", "content": error_msg, "sources": []}
                )
