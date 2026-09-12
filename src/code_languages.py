"""
Which file extensions are worth indexing, and how each should be split.

Lives in its own module because two unrelated callers need it and neither
should have to import the other: ``repo_ingestion`` uses it as the ingest
allowlist, ``data_ingestion`` uses it to choose a splitter. Keeping it here
means ``repo_ingestion`` never has to import ``data_ingestion`` (which pulls in
Streamlit, EasyOCR and PyMuPDF) just to know what a ``.py`` file is.

Only stdlib plus the Language enum, so it stays importable in tests.
"""

from __future__ import annotations

import os

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Extension -> LangChain Language, for syntax-aware splitting.
#
# Deliberately conservative: an extension appears here only when LangChain
# ships separators for the language. Anything absent still gets indexed via
# TEXT_EXTENSIONS or the generic splitter — the map controls split *quality*,
# not whether a file is ingested.
EXTENSION_LANGUAGE: dict[str, Language] = {
    ".py":    Language.PYTHON,
    ".js":    Language.JS,
    ".jsx":   Language.JS,
    ".mjs":   Language.JS,
    ".cjs":   Language.JS,
    ".ts":    Language.TS,
    ".tsx":   Language.TS,
    ".java":  Language.JAVA,
    ".kt":    Language.KOTLIN,
    ".go":    Language.GO,
    ".rs":    Language.RUST,
    ".rb":    Language.RUBY,
    ".php":   Language.PHP,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".c":     Language.C,
    ".h":     Language.C,
    ".cpp":   Language.CPP,
    ".cc":    Language.CPP,
    ".hpp":   Language.CPP,
    ".cs":    Language.CSHARP,
    ".lua":   Language.LUA,
    ".ex":    Language.ELIXIR,
    ".exs":   Language.ELIXIR,
    ".hs":    Language.HASKELL,
    ".sol":   Language.SOL,
    ".proto": Language.PROTO,
    ".html":  Language.HTML,
    ".md":    Language.MARKDOWN,
    ".rst":   Language.RST,
}

# Text formats worth indexing that have no language-specific separators.
# Config files are included because "what does the CI config do" is a real
# question; the lockfile and size filters in repo_ingestion keep the
# pathological cases (package-lock.json) out.
TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".sql", ".css", ".scss",
    ".gradle", ".dockerfile", ".tf", ".r", ".pl", ".vue", ".svelte",
})

# Extensionless files that are almost always worth reading.
BARE_FILENAMES: frozenset[str] = frozenset({
    "dockerfile", "makefile", "rakefile", "gemfile", "procfile",
    "readme", "license", "changelog", "codeowners",
})

INDEXABLE_EXTENSIONS: frozenset[str] = frozenset(EXTENSION_LANGUAGE) | TEXT_EXTENSIONS


def is_indexable(path: str) -> bool:
    """True if this path looks like readable source or documentation."""
    ext = os.path.splitext(path)[1].lower()
    if ext in INDEXABLE_EXTENSIONS:
        return True
    return os.path.basename(path).lower() in BARE_FILENAMES


def language_for_path(path: str) -> Language | None:
    """The Language to split this file with, or None for the generic splitter."""
    return EXTENSION_LANGUAGE.get(os.path.splitext(path)[1].lower())


def separators_for(language: Language) -> list[str] | None:
    """
    Separator list for a language, or None when this LangChain version does
    not support it.

    Not every member of the Language enum has separators — ``Language.PERL``
    is in the enum and raises ValueError on lookup in the installed version.
    Since the supported set varies across langchain-text-splitters releases,
    this is checked at runtime rather than assumed from the enum.
    """
    try:
        return RecursiveCharacterTextSplitter.get_separators_for_language(language)
    except (ValueError, KeyError):
        return None
