"""
Per-process bookkeeping for chat vector stores, and the policy that evicts
them to keep an instance inside its memory budget.

Why this exists
---------------
Every chat that indexes something gets its own Chroma store (see
store_paths.py), and nothing used to delete one. On Cloud Run the filesystem
lives in RAM, and each store also pins a cached Chroma client and a BM25 index
inside the process, so memory only grew until the instance was recycled or
killed for exceeding its limit.

Policy, applied whenever a store is created or grows
----------------------------------------------------
1. Idle: a store unused for ``idle_seconds`` is evicted. Streamlit gives no
   reliable signal that a visitor closed the tab, so inactivity stands in.
2. Budget: while there are more than ``max_count`` stores, or their files
   total more than ``max_mb``, the least-recently-used store is evicted.
3. The store the current request is using is never evicted, even if it alone
   exceeds the budget. Upload and repository size limits bound that case.

Deliberately stdlib-only, like store_paths.py: this code decides what gets
deleted, so it must be testable without Streamlit or Chroma. It only ever
evicts paths it registered itself and never scans the disk, so stores from
local development, evaluation runs or another process are never touched.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping

# Sized for the recommended Cloud Run instance (1 vCPU); see the measurement
# notes on StoreBudget. Each can be overridden with the env var named beside it.
DEFAULT_MAX_COUNT = 40          # RAG_STORE_MAX_COUNT
DEFAULT_MAX_MB = 512            # RAG_STORE_MAX_MB     (store files on disk)
DEFAULT_IDLE_MINUTES = 60       # RAG_STORE_IDLE_MINUTES


@dataclass(frozen=True)
class StoreBudget:
    """How many chat stores one process may keep, and for how long."""

    max_count: int = DEFAULT_MAX_COUNT
    max_mb: float = DEFAULT_MAX_MB
    idle_seconds: float = DEFAULT_IDLE_MINUTES * 60


def _positive(env: Mapping[str, str], name: str, default, cast):
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive number, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive number, got {raw!r}")
    return value


def budget_from_env(env: Mapping[str, str] | None = None) -> StoreBudget:
    """
    The budget, with overrides from the environment. Unset or blank variables
    keep the defaults; a malformed value raises, so a typo in the deployment
    config fails loudly at startup instead of silently disabling eviction.
    """
    env = os.environ if env is None else env
    idle_minutes = _positive(env, "RAG_STORE_IDLE_MINUTES", DEFAULT_IDLE_MINUTES, float)
    return StoreBudget(
        max_count=_positive(env, "RAG_STORE_MAX_COUNT", DEFAULT_MAX_COUNT, int),
        max_mb=_positive(env, "RAG_STORE_MAX_MB", DEFAULT_MAX_MB, float),
        idle_seconds=idle_minutes * 60,
    )


def dir_size_mb(path: str) -> float:
    """Total size of the files under ``path``, in MB. Missing paths count as 0."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue  # removed between listing and stat: no longer counts
    return total / (1024 * 1024)


class StoreRegistry:
    """
    The stores this process created, and when each was last used.

    Thread-safe: Streamlit runs every browser session as a thread of one
    process, so several sessions can register, touch and evict at once.
    ``release`` does the actual cleanup (closing caches, deleting files); it
    runs outside the lock so one slow deletion never blocks other sessions.
    """

    def __init__(
        self,
        budget: StoreBudget,
        release: Callable[[str], None],
        *,
        clock: Callable[[], float] = time.monotonic,
        size_mb: Callable[[str], float] = dir_size_mb,
    ) -> None:
        self._budget = budget
        self._release = release
        self._clock = clock
        self._size_mb = size_mb
        self._last_used: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def budget(self) -> StoreBudget:
        return self._budget

    def __contains__(self, path: object) -> bool:
        with self._lock:
            return path in self._last_used

    def __len__(self) -> int:
        with self._lock:
            return len(self._last_used)

    def register(self, path: str) -> None:
        """Start tracking a store this process just created."""
        with self._lock:
            self._last_used[path] = self._clock()

    def touch(self, path: str) -> None:
        """Mark a store as just used. Unknown paths are ignored, not adopted."""
        with self._lock:
            if path in self._last_used:
                self._last_used[path] = self._clock()

    def enforce(self, keep: str) -> list[str]:
        """
        Evict idle stores, then least-recently-used ones until the budget is
        met. ``keep`` (the store in use) is never evicted. Returns the paths
        evicted, oldest first.
        """
        with self._lock:
            victims = self._select_victims(keep)
            for path in victims:
                del self._last_used[path]

        for path in victims:
            try:
                self._release(path)
            except Exception as exc:  # one bad store must not fail the request
                print(f"[WARNING] Could not evict store '{path}': {exc}", file=sys.stderr)
        return victims

    def _select_victims(self, keep: str) -> list[str]:
        """Caller holds the lock."""
        now = self._clock()
        idle_after = self._budget.idle_seconds
        idle = [
            path for path, used in self._last_used.items()
            if path != keep and now - used > idle_after
        ]
        live = sorted(
            (used, path) for path, used in self._last_used.items() if path not in idle
        )
        sizes = {path: self._size_mb(path) for _used, path in live}
        count, total_mb = len(live), sum(sizes.values())

        over_budget = []
        for _used, path in live:             # oldest first
            if count <= self._budget.max_count and total_mb <= self._budget.max_mb:
                break
            if path == keep:
                continue
            over_budget.append(path)
            count -= 1
            total_mb -= sizes[path]
        return sorted(idle, key=self._last_used.__getitem__) + over_budget
