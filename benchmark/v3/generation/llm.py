"""
Cached, cost-tracked, throttled Gemini calls for the generation eval.

Every call is keyed by sha256(model, purpose, sample, prompt, config) and
stored in an append-only JSONL cache under benchmark/.cache/generation/
(gitignored: it holds prompts and answers). A rerun of the same eval makes
zero API calls and costs nothing.

* ``purpose`` separates call kinds ("answer", "judge", ...).
* ``sample`` > 0 deliberately bypasses an earlier cached result, for the
  noise runs that re-generate or re-judge the same input.

Cost is computed from the token counts Gemini returns (``usage_metadata``),
with thinking tokens billed as output. A run aborts before any call that
would take its new spend over ``max_usd``. Cached calls cost nothing.

The backend is injectable, so tests run with a fake and never reach the
network. Nothing here prints prompt or answer text.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

GENERATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(GENERATION_DIR)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "benchmark", ".cache", "generation")
CACHE_PATH = os.path.join(CACHE_DIR, "llm_cache.jsonl")

# USD per 1M tokens: (input, output incl. thinking). Standard paid-tier list
# prices from third-party listings of Google's rates, November 2026; Google's
# pricing page is the source of truth. gemini-3.7-flash is an introductory
# price, valid through 2026-12-31. Override with --price MODEL=IN,OUT.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.7-flash": (0.75, 3.75),
}

RETRYABLE_CODES = {429, 500, 503, 504}


class SpendCapReached(RuntimeError):
    """The next call would take this run's new spend over its cap."""


class CallFailed(RuntimeError):
    """A call kept failing after retries (quota, outage, unknown model)."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


@dataclass
class Usage:
    prompt: int = 0
    output: int = 0          # candidates
    thoughts: int = 0

    def cost(self, model: str, prices: dict) -> float:
        p_in, p_out = prices.get(model, (0.0, 0.0))
        return (self.prompt * p_in + (self.output + self.thoughts) * p_out) / 1_000_000


@dataclass
class Reply:
    """What callers get back. ``text`` is never logged."""
    text: str
    usage: Usage
    seconds: float
    cached: bool
    model: str


@dataclass
class Ledger:
    """Calls and spend for one run, per (model, purpose). No text."""
    new_calls: int = 0
    cached_calls: int = 0
    new_usd: float = 0.0
    by_key: dict = field(default_factory=dict)

    def add(self, model: str, purpose: str, reply: Reply, usd: float) -> None:
        row = self.by_key.setdefault(f"{model}|{purpose}", {
            "model": model, "purpose": purpose, "new_calls": 0, "cached_calls": 0,
            "prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0,
            "usd": 0.0, "seconds": 0.0})
        if reply.cached:
            self.cached_calls += 1
            row["cached_calls"] += 1
        else:
            self.new_calls += 1
            self.new_usd += usd
            row["new_calls"] += 1
            row["usd"] += usd
            row["seconds"] += reply.seconds
        row["prompt_tokens"] += reply.usage.prompt
        row["output_tokens"] += reply.usage.output
        row["thought_tokens"] += reply.usage.thoughts

    def summary(self) -> dict:
        return {"new_calls": self.new_calls, "cached_calls": self.cached_calls,
                "new_usd": round(self.new_usd, 5),
                "by_model_and_purpose": {k: {**v, "usd": round(v["usd"], 5),
                                             "seconds": round(v["seconds"], 2)}
                                         for k, v in sorted(self.by_key.items())}}


def cache_key(model: str, purpose: str, sample: int, prompt: str, config: dict | None) -> str:
    blob = json.dumps({"model": model, "purpose": purpose, "sample": sample,
                       "prompt": prompt, "config": config or {}},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    """Append-only JSONL; the last record for a key wins."""

    def __init__(self, path: str = CACHE_PATH):
        self.path = path
        self.records: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self.records[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def put(self, record: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records[record["key"]] = record


Backend = Callable[[str, str, dict | None], tuple[str, Usage]]


def gemini_backend(api_key: str) -> Backend:
    """The real backend: google-genai, with the given config (or none)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def call(model: str, prompt: str, config: dict | None) -> tuple[str, Usage]:
        kwargs: dict[str, Any] = {"model": model, "contents": prompt}
        if config:
            conf = dict(config)
            level = conf.pop("thinking_level", None)
            if level:
                conf["thinking_config"] = types.ThinkingConfig(thinking_level=level)
            kwargs["config"] = types.GenerateContentConfig(**conf)
        response = client.models.generate_content(**kwargs)
        meta = getattr(response, "usage_metadata", None)
        usage = Usage(
            prompt=int(getattr(meta, "prompt_token_count", 0) or 0),
            output=int(getattr(meta, "candidates_token_count", 0) or 0),
            thoughts=int(getattr(meta, "thoughts_token_count", 0) or 0),
        )
        return (getattr(response, "text", None) or ""), usage

    return call


def _error_code(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


class CachedLLM:
    """
    The one way the eval calls a model: cache, throttle, retry, cost, cap.
    """

    def __init__(self, backend: Backend | None, cache: Cache, *, max_usd: float,
                 rpm: float, prices: dict | None = None, retries: int = 4,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 estimate_usd: Callable[[str], float] | None = None):
        self.backend = backend
        self.cache = cache
        self.max_usd = max_usd
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self.prices = dict(PRICES if prices is None else prices)
        self.retries = retries
        self.sleep = sleep
        self.clock = clock
        self.estimate_usd = estimate_usd or (lambda model: 0.0)
        self.ledger = Ledger()
        self._last_call: float | None = None
        self.dropped_thinking: set[str] = set()

    def __call__(self, model: str, purpose: str, prompt: str, *,
                 config: dict | None = None, sample: int = 0) -> Reply:
        key = cache_key(model, purpose, sample, prompt, config)
        hit = self.cache.get(key)
        if hit is not None:
            u = hit["usage"]
            reply = Reply(hit["text"], Usage(u["prompt"], u["output"], u["thoughts"]),
                          hit["seconds"], cached=True, model=model)
            self.ledger.add(model, purpose, reply, 0.0)
            return reply

        if self.backend is None:
            raise CallFailed("no API key: this call is not cached and no backend is configured")
        if self.ledger.new_usd + self.estimate_usd(model) > self.max_usd:
            raise SpendCapReached(
                f"stopping before a new {model} call: this run has spent "
                f"${self.ledger.new_usd:.4f} of its ${self.max_usd:.2f} cap")

        text, usage, seconds = self._call_with_retries(model, prompt, config)
        usd = usage.cost(model, self.prices)
        reply = Reply(text, usage, seconds, cached=False, model=model)
        self.cache.put({"key": key, "model": model, "purpose": purpose, "sample": sample,
                        "text": text, "usage": usage.__dict__, "seconds": round(seconds, 3),
                        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self.ledger.add(model, purpose, reply, usd)
        return reply

    def _throttle(self) -> None:
        if self._last_call is not None and self.min_interval:
            wait = self.min_interval - (self.clock() - self._last_call)
            if wait > 0:
                self.sleep(wait)
        self._last_call = self.clock()

    def _call_with_retries(self, model: str, prompt: str, config: dict | None):
        delay = 5.0
        for attempt in range(self.retries + 1):
            self._throttle()
            started = self.clock()
            try:
                text, usage = self.backend(model, prompt, config)
                return text, usage, self.clock() - started
            except Exception as exc:  # classified below: retried, adapted or reported
                code = _error_code(exc)
                if (code == 400 and config and "thinking_level" in config
                        and "thinking" in str(exc).lower()):
                    # This model does not accept the thinking setting; retry
                    # without it once and remember that, for the report.
                    config = {k: v for k, v in config.items() if k != "thinking_level"}
                    self.dropped_thinking.add(model)
                    continue
                if code in RETRYABLE_CODES and attempt < self.retries:
                    print(f"[gen-eval] {model}: HTTP {code}, retrying in {delay:.0f}s "
                          f"({attempt + 1}/{self.retries})", file=sys.stderr, flush=True)
                    self.sleep(delay)
                    delay *= 2
                    continue
                raise CallFailed(f"{model}: {type(exc).__name__} (HTTP {code})", code) from exc
        raise CallFailed(f"{model}: gave up after {self.retries} retries")


def bind(llm: CachedLLM, model: str, purpose: str, *, sample: int = 0,
         config: dict | None = None, on_reply: Callable[[Reply], None] | None = None):
    """A ``prompt -> object with .text`` callable, the shape build_rag_chain takes."""

    def call(prompt: str) -> Reply:
        reply = llm(model, purpose, prompt, config=config, sample=sample)
        if on_reply is not None:
            on_reply(reply)
        return reply

    return call
