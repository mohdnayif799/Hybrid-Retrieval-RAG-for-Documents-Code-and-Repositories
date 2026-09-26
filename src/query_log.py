"""
One log line per query, and classification of Gemini failures.

Same shape as AgentCrew's run log: status, which key was used (never the key
itself), model, duration and number of Gemini calls. The question text and the
key are never logged; the logging function does not even accept them, so
neither can leak by accident. Lines go to stderr, which Cloud Run collects
into Cloud Logging.
"""

from __future__ import annotations

import sys
from typing import TextIO

QUOTA_STATUS = "RESOURCE_EXHAUSTED"


class CallCounter:
    """Counts Gemini calls. Passed into the RAG chain and called before each one."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> None:
        self.count += 1


def is_quota_error(exc: BaseException) -> bool:
    """
    True for Gemini's quota / rate-limit refusal: HTTP 429, RESOURCE_EXHAUSTED.

    Reads the attributes google-genai's APIError carries (``code``, ``status``)
    rather than matching message text, which changes between SDK versions.
    """
    return getattr(exc, "code", None) == 429 or getattr(exc, "status", None) == QUOTA_STATUS


def redact(text: str, secret: str | None) -> str:
    """``text`` with ``secret`` blanked out, for echoing a visitor's own error."""
    secret = (secret or "").strip()
    return text.replace(secret, "[redacted]") if secret else text


def format_query_log(status: str, key: str, model: str, seconds: float,
                     gemini_calls: int, error: BaseException | None = None) -> str:
    """
    The log line. ``key`` is "shared" or "own", never a key value. On failure
    only the exception's type and HTTP status code are included: enough to
    tell a quota refusal from a wrong model name (404) or an outage (5xx),
    without provider message text that could echo request content.
    """
    line = (
        f"query status={status} key={key} model={model} "
        f"seconds={seconds:.1f} gemini_calls={gemini_calls}"
    )
    if error is None:
        return line
    line = f"{line} error={type(error).__name__}"
    code = getattr(error, "code", None)
    return f"{line} code={code}" if isinstance(code, int) else line


def log_query(status: str, key: str, model: str, seconds: float,
              gemini_calls: int, error: BaseException | None = None,
              stream: TextIO | None = None) -> None:
    """Write one query's log line to stderr."""
    print(format_query_log(status, key, model, seconds, gemini_calls, error),
          file=stream or sys.stderr, flush=True)
