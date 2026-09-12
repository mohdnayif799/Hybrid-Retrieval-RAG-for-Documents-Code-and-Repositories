"""
Naming and lifecycle rules for Chroma persistence directories.

Deliberately dependency-free (stdlib only). Two reasons:

1. ``vector_store`` imports Streamlit, HuggingFace embeddings and Chroma, so
   anything living there can only be tested with the full ML stack installed.
   The rules below decide which directories get *deleted*, which makes them
   the most safety-critical code in the project and the code that most needs
   cheap, fast tests.
2. It keeps the "which directory belongs to whom" question in one place.

The namespacing rule
--------------------
Two independent producers create stores:

  * ``app.py``      -> per-chat stores, one per Streamlit chat, possibly several
                      alive at once, and live for as long as the app runs.
  * ``evaluate.py`` -> a throwaway store per evaluation run, which wants to
                      clean up after previous runs.

Previously both used the ``chroma_db`` prefix and the eval cleanup globbed
``chroma_db*``, so running an evaluation deleted every open chat's store. The
app kept running with a ``chroma_dir`` pointing at nothing and silently
answered "I don't have enough information" from then on.

The fix is structural rather than a runtime guard: eval stores now live under
a *different prefix*, and the cleanup glob is scoped to that prefix. The
destructive operation can no longer name chat data, so no amount of future
refactoring in ``evaluate.py`` can reintroduce the bug.
"""

from __future__ import annotations

import glob
import os
import time
import uuid

# Directory prefixes. Neither may be a prefix of the other, or the globs below
# would overlap and the isolation guarantee would silently disappear.
CHAT_STORE_PREFIX = "chroma_db"
EVAL_STORE_PREFIX = "chroma_eval"

assert not CHAT_STORE_PREFIX.startswith(EVAL_STORE_PREFIX)
assert not EVAL_STORE_PREFIX.startswith(CHAT_STORE_PREFIX)


def project_root() -> str:
    """Absolute path of the repository root (the parent of ``src/``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _unique_suffix() -> str:
    """
    Timestamp plus a short random suffix.

    A bare ``int(time.time())`` collides when two stores are created inside the
    same second — two chats processed back to back would share one directory
    and silently merge their documents.
    """
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}"


def new_chat_store_dir(root: str | None = None) -> str:
    """Path for a new per-chat store. Never deleted by the eval cleanup."""
    return os.path.join(root or project_root(), f"{CHAT_STORE_PREFIX}_{_unique_suffix()}")


def new_eval_store_dir(root: str | None = None) -> str:
    """Path for a new evaluation store. Subject to the eval cleanup."""
    return os.path.join(root or project_root(), f"{EVAL_STORE_PREFIX}_{_unique_suffix()}")


def is_chat_store_dir(path: str) -> bool:
    return os.path.basename(os.path.normpath(path)).startswith(CHAT_STORE_PREFIX + "_")


def is_eval_store_dir(path: str) -> bool:
    return os.path.basename(os.path.normpath(path)).startswith(EVAL_STORE_PREFIX + "_")


def stale_eval_store_dirs(keep: str, root: str | None = None) -> list[str]:
    """
    Evaluation stores that may safely be deleted: every ``chroma_eval_*``
    directory except ``keep``.

    Chat stores are excluded twice over — the glob cannot match them, and
    ``is_eval_store_dir`` re-checks every candidate. The redundancy is
    deliberate: this function's output is fed straight to ``shutil.rmtree``,
    so a second cheap check is worth more than the line it costs.
    """
    base = root or project_root()
    keep_real = os.path.normpath(os.path.abspath(keep))

    stale = []
    for path in glob.glob(os.path.join(base, f"{EVAL_STORE_PREFIX}_*")):
        if not os.path.isdir(path):
            continue
        if not is_eval_store_dir(path):        # defence in depth
            continue
        if os.path.normpath(os.path.abspath(path)) == keep_real:
            continue
        stale.append(path)
    return sorted(stale)
