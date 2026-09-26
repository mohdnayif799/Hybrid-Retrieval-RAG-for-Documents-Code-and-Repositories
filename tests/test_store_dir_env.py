"""RAG_STORE_DIR moves per-chat stores; unset, they stay in the project root."""

import os

from src.store_paths import CHAT_STORE_PREFIX, chat_store_root, new_chat_store_dir, project_root


def test_chat_stores_default_to_the_project_root(monkeypatch):
    monkeypatch.delenv("RAG_STORE_DIR", raising=False)
    assert chat_store_root() == project_root()


def test_blank_rag_store_dir_is_ignored(monkeypatch):
    monkeypatch.setenv("RAG_STORE_DIR", "   ")
    assert chat_store_root() == project_root()


def test_rag_store_dir_moves_new_chat_stores(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_STORE_DIR", str(tmp_path))
    path = new_chat_store_dir()
    assert os.path.dirname(path) == str(tmp_path)
    assert os.path.basename(path).startswith(CHAT_STORE_PREFIX + "_")
