"""
Cleanup isolation tests.

The bug these guard against: evaluate.py's store cleanup globbed ``chroma_db*``
and deleted every directory except the one it had just created — including the
live per-chat stores belonging to a running Streamlit app. The app kept its
now-dangling ``chroma_dir`` in session state and answered "I don't have enough
information" from then on, with no error anywhere.

These run with stdlib only.
"""

import os

from src.store_paths import (
    CHAT_STORE_PREFIX,
    EVAL_STORE_PREFIX,
    is_chat_store_dir,
    is_eval_store_dir,
    new_chat_store_dir,
    new_eval_store_dir,
    stale_eval_store_dirs,
)


def _make(tmp_path, *names):
    for name in names:
        (tmp_path / name).mkdir()
    return str(tmp_path)


# ── The core guarantee ────────────────────────────────────────────────────────

def test_eval_cleanup_never_returns_chat_stores(tmp_path):
    root = _make(
        tmp_path,
        "chroma_db_1000_aaaaaaaa",     # live chat store
        "chroma_db_2000_bbbbbbbb",     # live chat store
        "chroma_eval_3000_cccccccc",   # stale eval store
        "chroma_eval_4000_dddddddd",   # the run we are keeping
    )
    keep = os.path.join(root, "chroma_eval_4000_dddddddd")

    stale = stale_eval_store_dirs(keep, root=root)

    assert stale == [os.path.join(root, "chroma_eval_3000_cccccccc")]
    assert not any(is_chat_store_dir(p) for p in stale), "a chat store was marked for deletion"


def test_keep_directory_is_never_stale(tmp_path):
    root = _make(tmp_path, "chroma_eval_1000_aaaaaaaa")
    keep = os.path.join(root, "chroma_eval_1000_aaaaaaaa")
    assert stale_eval_store_dirs(keep, root=root) == []


def test_cleanup_is_a_noop_when_only_chat_stores_exist(tmp_path):
    """The exact scenario from the bug: eval run while chats are open."""
    root = _make(tmp_path, "chroma_db_1_aaaaaaaa", "chroma_db_2_bbbbbbbb", "chroma_db_3_cccccccc")
    keep = os.path.join(root, "chroma_eval_9_zzzzzzzz")   # not created yet
    assert stale_eval_store_dirs(keep, root=root) == []


def test_unrelated_directories_are_ignored(tmp_path):
    root = _make(tmp_path, "chroma_evaluation_notes", "chromadb_backup", "src", "docs")
    stale = stale_eval_store_dirs(os.path.join(root, "chroma_eval_1_a"), root=root)
    assert stale == [], f"unexpected deletion candidates: {stale}"


def test_files_are_not_returned_only_directories(tmp_path):
    (tmp_path / "chroma_eval_1000_aaaaaaaa").write_text("a file, not a store")
    assert stale_eval_store_dirs(os.path.join(str(tmp_path), "keep"), root=str(tmp_path)) == []


# ── Naming ────────────────────────────────────────────────────────────────────

def test_prefixes_cannot_shadow_each_other():
    # If one prefix were a prefix of the other, the eval glob would match chat
    # stores and the isolation guarantee would vanish silently.
    assert not CHAT_STORE_PREFIX.startswith(EVAL_STORE_PREFIX)
    assert not EVAL_STORE_PREFIX.startswith(CHAT_STORE_PREFIX)


def test_chat_and_eval_dirs_are_classified_correctly(tmp_path):
    chat = new_chat_store_dir(root=str(tmp_path))
    ev = new_eval_store_dir(root=str(tmp_path))
    assert is_chat_store_dir(chat) and not is_eval_store_dir(chat)
    assert is_eval_store_dir(ev) and not is_chat_store_dir(ev)


def test_store_dirs_do_not_collide_within_the_same_second(tmp_path):
    # int(time.time()) alone collides for back-to-back ingestions, which would
    # silently merge two chats' documents into one store.
    names = {new_chat_store_dir(root=str(tmp_path)) for _ in range(50)}
    assert len(names) == 50
