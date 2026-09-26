"""Eviction policy for chat stores: src/store_registry.py (stdlib only)."""

import pytest

from src.store_registry import StoreBudget, StoreRegistry, budget_from_env, dir_size_mb

NEVER_IDLE = 1e9


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _registry(budget, sizes=None):
    clock, released = FakeClock(), []
    registry = StoreRegistry(
        budget,
        release=released.append,
        clock=clock,
        size_mb=lambda path: (sizes or {}).get(path, 1.0),
    )
    return registry, clock, released


def _register_in_order(registry, clock, paths):
    for tick, path in enumerate(paths):
        clock.now = tick
        registry.register(path)


def test_store_unused_past_the_idle_limit_is_evicted():
    registry, clock, released = _registry(StoreBudget(10, 100, idle_seconds=60))
    registry.register("a")
    clock.now = 30
    registry.register("b")

    clock.now = 61      # "a" idle for 61 s, "b" for 31 s
    assert registry.enforce(keep="b") == ["a"]
    assert released == ["a"]
    assert "a" not in registry and "b" in registry


def test_touch_resets_the_idle_timer():
    registry, clock, released = _registry(StoreBudget(10, 100, idle_seconds=60))
    registry.register("a")
    clock.now = 50
    registry.touch("a")
    registry.register("b")

    clock.now = 100     # "a" last used 50 s ago
    assert registry.enforce(keep="b") == []
    assert released == []


def test_count_budget_evicts_the_least_recently_used_first():
    registry, clock, _ = _registry(StoreBudget(max_count=2, max_mb=100, idle_seconds=NEVER_IDLE))
    _register_in_order(registry, clock, ["a", "b", "c"])
    clock.now = 5
    registry.touch("a")     # "b" is now the least recently used

    assert registry.enforce(keep="c") == ["b"]
    assert len(registry) == 2


def test_size_budget_evicts_until_the_total_fits():
    sizes = {"a": 300, "b": 300, "c": 300}
    registry, clock, _ = _registry(StoreBudget(max_count=10, max_mb=500, idle_seconds=NEVER_IDLE), sizes)
    _register_in_order(registry, clock, ["a", "b", "c"])

    assert registry.enforce(keep="c") == ["a", "b"]     # 900 -> 600 -> 300 MB


def test_store_in_use_is_never_evicted_even_when_it_alone_is_over_budget():
    registry, clock, released = _registry(StoreBudget(max_count=1, max_mb=10, idle_seconds=1), {"big": 5000})
    registry.register("big")
    clock.now = 100

    assert registry.enforce(keep="big") == []
    assert released == [] and "big" in registry


def test_touch_never_adopts_a_path_it_did_not_register():
    registry, _, _ = _registry(StoreBudget())
    registry.touch("/somebody/elses/store")
    assert "/somebody/elses/store" not in registry
    assert len(registry) == 0


def test_a_failing_release_is_logged_and_does_not_stop_the_others(capsys):
    released = []

    def release(path):
        if path == "a":
            raise OSError("device busy")
        released.append(path)

    clock = FakeClock()
    registry = StoreRegistry(StoreBudget(max_count=1, max_mb=100, idle_seconds=NEVER_IDLE),
                             release=release, clock=clock, size_mb=lambda path: 1.0)
    _register_in_order(registry, clock, ["a", "b", "c"])

    assert registry.enforce(keep="c") == ["a", "b"]
    assert released == ["b"]
    assert "Could not evict store 'a'" in capsys.readouterr().err


def test_budget_from_env_keeps_defaults_when_unset_or_blank():
    assert budget_from_env({}) == StoreBudget()
    assert budget_from_env({"RAG_STORE_MAX_COUNT": "  "}) == StoreBudget()


def test_budget_from_env_reads_overrides():
    budget = budget_from_env({
        "RAG_STORE_MAX_COUNT": "5",
        "RAG_STORE_MAX_MB": "256.5",
        "RAG_STORE_IDLE_MINUTES": "2",
    })
    assert budget == StoreBudget(max_count=5, max_mb=256.5, idle_seconds=120)


@pytest.mark.parametrize("name,value", [
    ("RAG_STORE_MAX_COUNT", "0"),
    ("RAG_STORE_MAX_COUNT", "ten"),
    ("RAG_STORE_MAX_MB", "-1"),
    ("RAG_STORE_IDLE_MINUTES", "soon"),
])
def test_budget_from_env_rejects_malformed_values(name, value):
    with pytest.raises(ValueError, match=name):
        budget_from_env({name: value})


def test_dir_size_mb_counts_nested_files_and_treats_missing_as_empty(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.bin").write_bytes(b"x" * 1024 * 1024)
    (tmp_path / "b.bin").write_bytes(b"x" * 512 * 1024)

    assert dir_size_mb(str(tmp_path)) == pytest.approx(1.5)
    assert dir_size_mb(str(tmp_path / "missing")) == 0
