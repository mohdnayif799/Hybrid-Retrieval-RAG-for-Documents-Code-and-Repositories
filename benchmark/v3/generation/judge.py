"""
The LLM judge: prompt assembly, output schema, parsing, derived verdicts.

The prompt and rubric live in ``judge_v1.md``. Its SHA-256 is pinned in
``manifest.json``; ``verify_judge()`` refuses to grade if the file changed
without a new version (a changed rubric means judge_v2.md and a re-judge,
which costs little because answers are cached).
"""

from __future__ import annotations

import hashlib
import json
import os
import re

GENERATION_DIR = os.path.dirname(os.path.abspath(__file__))
JUDGE_VERSION = "v1"
JUDGE_PATH = os.path.join(GENERATION_DIR, f"judge_{JUDGE_VERSION}.md")
MANIFEST_PATH = os.path.join(GENERATION_DIR, "manifest.json")

# Same exact sentence as src/rag_chain._QA_TEMPLATE asks for.
REFUSAL_SENTENCE = "I don't have enough information in the provided documents to answer this."

MAX_SPAN_CHARS = 1500       # long gold sections are cut, keeping the key fact
CONTEXT_SEPARATOR = "\n\n"

FIELDS = {
    "refused": ("yes", "no"),
    "correct": ("yes", "partial", "no", "na"),
    "grounded": ("yes", "no", "na"),
    "length_fit": ("too_short", "right", "too_long"),
}

SCHEMA = {
    "type": "object",
    "properties": {
        "refused": {"type": "string", "enum": list(FIELDS["refused"])},
        "correct": {"type": "string", "enum": list(FIELDS["correct"])},
        "grounded": {"type": "string", "enum": list(FIELDS["grounded"])},
        "unsupported_claims": {"type": "integer", "minimum": 0},
        "length_fit": {"type": "string", "enum": list(FIELDS["length_fit"])},
        "notes": {"type": "string"},
    },
    "required": ["refused", "correct", "grounded", "unsupported_claims", "length_fit", "notes"],
    "propertyOrdering": ["refused", "correct", "grounded", "unsupported_claims",
                         "length_fit", "notes"],
}

JUDGE_CONFIG = {
    "temperature": 0.0,
    "response_mime_type": "application/json",
    "response_json_schema": SCHEMA,
    "thinking_level": "LOW",
}


class JudgeError(ValueError):
    pass


def sha256_file(path: str) -> str:
    """CRLF read as LF, so a Windows checkout verifies the same (see freeze.py)."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()


def load_template(path: str = JUDGE_PATH) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def verify_judge(manifest_path: str = MANIFEST_PATH, judge_path: str = JUDGE_PATH) -> str:
    """The pinned hash of the judge prompt, or SystemExit if the file changed."""
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    name = os.path.basename(judge_path)
    pinned = manifest["judge"].get(name)
    actual = sha256_file(judge_path)
    if pinned != actual:
        raise SystemExit(
            f"refusing to grade: {name} changed since it was pinned in manifest.json "
            f"({(pinned or 'missing')[:12]} != {actual[:12]}). A rubric change needs a new "
            "version (judge_v2.md) and a manifest entry, so old and new grades never mix.")
    return actual


def _cut(text: str, keep: str | None, limit: int = MAX_SPAN_CHARS) -> str:
    """At most ``limit`` characters, centred on ``keep`` when it is given."""
    if len(text) <= limit:
        return text
    at = text.casefold().find((keep or "").casefold()) if keep else -1
    start = 0 if at < 0 else max(0, min(at - limit // 3, len(text) - limit))
    return ("[...] " if start else "") + text[start:start + limit] + " [...]"


def format_gold(spans: list, parents: dict, slice_fn) -> str:
    """Gold spans grouped by part, each with its key fact."""
    if not spans:
        return "(none: this question is not answerable from the documents)"
    by_part: dict[str, list] = {}
    for span in spans:
        for part in span.parts:
            by_part.setdefault(part, []).append(span)
    blocks = []
    for n, (part, members) in enumerate(sorted(by_part.items()), 1):
        lines = [f"PART {n}" + (" (alternatives)" if len(members) > 1 else "")]
        for span in members:
            paged = span.file.lower().endswith((".pdf", ".pptx"))
            where = span.file + (f" page {span.page + 1}" if paged and span.page is not None else "")
            lines.append(f"- from {where}; KEY FACT: {span.evidence or '(the whole passage)'}")
            lines.append("  " + _cut(slice_fn(parents, span), span.evidence).replace("\n", "\n  "))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_context(texts: list[str]) -> str:
    if not texts:
        return "(nothing was retrieved)"
    return CONTEXT_SEPARATOR.join(f"[{i}] {t}" for i, t in enumerate(texts, 1))


def build_prompt(template: str, *, question: str, answerable: bool, gold: str,
                 context: str, answer: str) -> str:
    return template.format(question=question, answerable="yes" if answerable else "no",
                           gold=gold, context=context, answer=answer)


def parse(text: str) -> dict:
    """Validate the judge's JSON. Raises JudgeError on anything malformed."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge output is not JSON ({exc.msg})") from exc
    out = {}
    for name, allowed in FIELDS.items():
        value = str(data.get(name, "")).strip().lower()
        if value not in allowed:
            raise JudgeError(f"judge field {name}={value!r} not in {allowed}")
        out[name] = value
    try:
        out["unsupported_claims"] = max(0, int(data.get("unsupported_claims", 0)))
    except (TypeError, ValueError) as exc:
        raise JudgeError("unsupported_claims is not an integer") from exc
    return out


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def is_exact_refusal(answer: str) -> bool:
    return REFUSAL_SENTENCE.casefold() in re.sub(r"\s+", " ", answer).casefold()


def derive(verdict: dict, answerable: bool, answer: str) -> dict:
    """Verdict plus fields computed in code (never by the judge)."""
    refused = verdict["refused"] == "yes"
    return {
        **verdict,
        # Correctness is only defined for answerable questions.
        "correct": verdict["correct"] if answerable else "na",
        "refusal_exact": is_exact_refusal(answer),
        "refusal_correct": (("yes" if refused else "no") if not answerable else "na"),
        "wrongly_refused": (refused if answerable else None),
        "word_count": word_count(answer),
    }
