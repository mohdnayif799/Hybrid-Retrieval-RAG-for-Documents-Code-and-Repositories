"""
Ingest a public GitHub repository as RAG source material.

Kept separate from ``data_ingestion`` on purpose. That module's contract is
*filesystem path -> Document*; this one adds networking, archive handling and
untrusted input, which is a different problem. More concretely: this module
never needs a LangChain loader, because it reads text straight out of the tar
stream. Routing repositories through ``load_documents`` would have required
writing every archive entry to disk first — the exact step that creates
path-traversal risk.

Security model
--------------
1. **No extraction to disk.** Members are streamed and read into memory.
   Nothing attacker-controlled is ever used as a write path, so zip-slip,
   symlink escapes and absolute-path writes are structurally impossible rather
   than merely mitigated. (Also why Python 3.12's ``extractall(filter=...)``
   is not needed — this runs on 3.10.)
2. **Regular files only.** ``member.isfile()`` rejects symlinks, hardlinks,
   directories, devices and FIFOs before anything is read.
3. **Paths still validated.** Even though nothing is written, a member path
   becomes citation metadata, so absolute paths and ``..`` segments are
   rejected rather than displayed.
4. **No user-controlled hosts.** Only ``api.github.com`` is ever contacted;
   owner and repo names are validated against a strict charset before being
   interpolated, so a crafted URL cannot redirect the request elsewhere (SSRF).
5. **Bounded work.** Declared uncompressed size is summed from tar headers
   *before* any member is read, so a decompression bomb is refused rather than
   buffered. Per-file reads are capped, as are file count and total bytes.

Failure policy: oversized repositories raise rather than being silently
truncated. A partially indexed repository is worse than a refused one — it
answers "I don't have enough information" for content the user believes is
indexed, with nothing anywhere indicating why.
"""

from __future__ import annotations

import io
import posixpath
import re
import tarfile
from dataclasses import dataclass, field
from typing import BinaryIO, Iterable
from urllib.parse import quote, urlparse

import requests
from langchain_core.documents import Document

from src.code_languages import is_indexable

GITHUB_API = "https://api.github.com"
GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})
SOURCE_TYPE_REPO = "repo"

HTTP_TIMEOUT = 30          # seconds, per request
DOWNLOAD_CHUNK = 64 * 1024


# ── Limits ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RepoLimits:
    """
    Ingestion ceilings. Chosen to admit ordinary project repositories while
    refusing monorepos and archive bombs. Heuristics, not measured optima.
    """
    max_repo_kb: int = 200 * 1024                  # 200 MB, from the API's size field
    max_archive_bytes: int = 150 * 1024 * 1024     # compressed download cap
    # Bomb guard on the whole archive walk. Counts the declared size of EVERY
    # member, including ones the filters drop, because skipping a member still
    # means decompressing past it. Set well above the download cap on purpose:
    # GitHub's `size` field is the PACKED repository size, and a working tree
    # routinely expands several times that, so a tight value here would refuse
    # legitimate repositories mid-download — and repositories with large
    # checked-in assets even though the indexable part is small. Raising it
    # does not raise peak memory: the download is capped at
    # max_archive_bytes, each member read is capped at max_file_bytes, and
    # retained text is capped at max_total_text_bytes. It only bounds time.
    max_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024
    max_file_bytes: int = 512 * 1024               # per file
    max_files: int = 3000                          # indexable files
    max_total_text_bytes: int = 40 * 1024 * 1024   # text actually kept
    max_line_length: int = 2000                    # longer => generated/minified


DEFAULT_LIMITS = RepoLimits()


# ── Errors ────────────────────────────────────────────────────────────────────

class RepositoryError(Exception):
    """Base class for every user-facing repository ingestion failure."""


class InvalidRepositoryURL(RepositoryError):
    pass


class RepositoryNotFound(RepositoryError):
    pass


class RepositoryTooLarge(RepositoryError):
    pass


class RepositoryEmpty(RepositoryError):
    pass


# ── URL parsing ───────────────────────────────────────────────────────────────

# GitHub owners are alphanumeric plus hyphen; repo names also allow "." and
# "_". The strictness is load-bearing: these values are interpolated into an
# api.github.com URL, so anything permitting "/" or "%" could redirect the
# request to another host.
_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_REF_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str
    ref: str | None = None

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    def label(self, ref: str | None = None) -> str:
        return f"{self.slug}@{ref or self.ref or 'default'}"


def parse_repo_url(url: str) -> RepoRef:
    """
    Parse a GitHub repository URL into owner / name / optional ref.

    Accepts ``https://github.com/owner/repo``, a trailing ``.git`` or slash,
    ``/tree/<branch>`` (including branch names containing slashes), a bare
    ``github.com/owner/repo``, and the ``owner/repo`` shorthand.
    """
    if not url or not url.strip():
        raise InvalidRepositoryURL("Enter a GitHub repository URL.")

    raw = url.strip()

    # Bare "owner/repo" shorthand.
    if "://" not in raw and "." not in raw.split("/")[0]:
        parts = [p for p in raw.split("/") if p]
        if len(parts) == 2:
            return _validated(parts[0], parts[1], None)
        raise InvalidRepositoryURL(f"Could not read a repository from {url!r}.")

    if "://" not in raw:
        raw = "https://" + raw

    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if host not in GITHUB_HOSTS:
        raise InvalidRepositoryURL(
            f"Only github.com repositories are supported (got {host or 'no host'})."
        )

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise InvalidRepositoryURL(
            "URL must include an owner and a repository, e.g. "
            "https://github.com/owner/repo"
        )

    owner, name = parts[0], parts[1]
    if name.endswith(".git"):
        name = name[: -len(".git")]

    ref = None
    if len(parts) > 3 and parts[2] in {"tree", "blob"}:
        ref = "/".join(parts[3:])       # branch names may contain slashes

    return _validated(owner, name, ref)


def _validated(owner: str, name: str, ref: str | None) -> RepoRef:
    if not _OWNER_RE.match(owner) or owner in {".", ".."}:
        raise InvalidRepositoryURL(f"Invalid repository owner: {owner!r}")
    if not _REPO_RE.match(name) or name in {".", ".."}:
        raise InvalidRepositoryURL(f"Invalid repository name: {name!r}")
    if ref is not None:
        if not _REF_RE.match(ref) or ".." in ref or ref.startswith("/"):
            raise InvalidRepositoryURL(f"Invalid branch or tag: {ref!r}")
    return RepoRef(owner=owner, name=name, ref=ref)


# ── Filtering rules ───────────────────────────────────────────────────────────

# Any path containing one of these as a whole component is dropped. These are
# the directories that make an unfiltered repository 10-50x larger than its
# useful content.
EXCLUDED_DIR_SEGMENTS: frozenset[str] = frozenset({
    ".git", "node_modules", "bower_components",
    "venv", ".venv", "env", "virtualenv", "site-packages", ".tox", ".nox",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "out", "target", "obj", ".next", ".nuxt",
    "vendor", "third_party", "coverage", "htmlcov", ".idea", ".vscode",
    ".gradle", ".terraform", "Pods", "DerivedData", ".cache", "__snapshots__",
})

EXCLUDED_FILENAMES: frozenset[str] = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "cargo.lock", "gemfile.lock", "composer.lock", "go.sum",
    "pipfile.lock", "bun.lockb", ".ds_store",
})

EXCLUDED_SUFFIXES: tuple[str, ...] = (
    ".lock", ".min.js", ".min.css", ".map", ".snap",
    ".bundle.js", ".chunk.js", ".pyc", ".pyo", ".so", ".dll", ".dylib",
    ".class", ".jar", ".war", ".exe", ".bin", ".o", ".a",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".pdf",
    ".zip", ".gz", ".tar", ".7z", ".rar", ".mp4", ".mp3", ".wav",
    ".woff", ".woff2", ".ttf", ".eot", ".parquet", ".db", ".sqlite",
)


class SkipReason:
    NOT_A_FILE = "not a regular file"
    UNSAFE_PATH = "unsafe path"
    EXCLUDED_DIR = "excluded directory"
    EXCLUDED_NAME = "excluded filename"
    NOT_INDEXABLE = "unsupported extension"
    TOO_LARGE = "file too large"
    BINARY = "not valid UTF-8"
    GENERATED = "generated or minified"
    EMPTY = "empty file"


@dataclass
class IngestReport:
    """Counts of what was kept and why anything was dropped. Shown in the UI."""
    kept: int = 0
    total_bytes: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    @property
    def skipped_total(self) -> int:
        return sum(self.skipped.values())


@dataclass(frozen=True)
class RepoFile:
    path: str      # repository-relative, POSIX separators
    text: str


def safe_member_path(name: str, strip_root: bool = True) -> str | None:
    """
    Normalise an archive member path, or None if it is unsafe.

    GitHub tarballs nest everything under ``<owner>-<repo>-<sha>/``, which is
    stripped. Absolute paths, drive letters and any ``..`` component are
    rejected — not because we would write them (we never do) but because the
    path is displayed to the user as a citation.
    """
    if not name:
        return None

    candidate = name.replace("\\", "/")          # Windows-style separators

    if candidate.startswith("/") or re.match(r"^[A-Za-z]:", candidate):
        return None

    if strip_root:
        head, _, tail = candidate.partition("/")
        candidate = tail if tail else ""

    if not candidate:
        return None

    parts = [p for p in candidate.split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        return None
    if not parts:
        return None

    normalised = posixpath.join(*parts)
    # posixpath.normpath as a final belt-and-braces check.
    if posixpath.normpath(normalised) != normalised:
        return None
    return normalised


def _is_excluded(path: str) -> str | None:
    """Return a SkipReason if this path should be dropped, else None."""
    parts = path.split("/")
    if any(part in EXCLUDED_DIR_SEGMENTS for part in parts[:-1]):
        return SkipReason.EXCLUDED_DIR

    filename = parts[-1].lower()
    if filename in EXCLUDED_FILENAMES:
        return SkipReason.EXCLUDED_NAME
    if filename.endswith(EXCLUDED_SUFFIXES):
        return SkipReason.EXCLUDED_NAME
    if not is_indexable(path):
        return SkipReason.NOT_INDEXABLE
    return None


def _looks_generated(text: str, max_line_length: int) -> bool:
    """Single enormous lines mean minified or generated output."""
    return any(len(line) > max_line_length for line in text.splitlines())


# ── Archive reading (pure — no network, testable with a fixture tarball) ──────

def read_archive(
    fileobj: BinaryIO,
    limits: RepoLimits = DEFAULT_LIMITS,
    progress_callback=None,
) -> tuple[list[RepoFile], IngestReport]:
    """
    Stream a ``.tar.gz`` and return the files worth indexing.

    Never writes to disk. Raises RepositoryTooLarge if the archive declares
    more uncompressed data than allowed, if too many indexable files are
    found, or if the retained text exceeds the cap.
    """
    files: list[RepoFile] = []
    report = IngestReport()
    declared_total = 0

    with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
        for member in tar:
            # Sum the header-declared size for EVERY member, including ones we
            # will skip. This catches a bomb before a single byte is read.
            declared_total += max(member.size, 0)
            if declared_total > limits.max_uncompressed_bytes:
                raise RepositoryTooLarge(
                    "Repository archive expands to more than "
                    f"{limits.max_uncompressed_bytes // (1024 * 1024)} MB — refusing to "
                    "process it. This is an archive-bomb guard, not a source-size limit."
                )

            if not member.isfile():
                continue                       # symlinks, dirs, devices, hardlinks

            path = safe_member_path(member.name)
            if path is None:
                report.skip(SkipReason.UNSAFE_PATH)
                continue

            reason = _is_excluded(path)
            if reason:
                report.skip(reason)
                continue

            if member.size > limits.max_file_bytes:
                report.skip(SkipReason.TOO_LARGE)
                continue

            handle = tar.extractfile(member)
            if handle is None:
                report.skip(SkipReason.NOT_A_FILE)
                continue

            # Bounded read: never buffer more than the per-file cap even if the
            # header under-reported the true size.
            raw = handle.read(limits.max_file_bytes + 1)
            if len(raw) > limits.max_file_bytes:
                report.skip(SkipReason.TOO_LARGE)
                continue

            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                report.skip(SkipReason.BINARY)     # binary with a text extension
                continue

            if not text.strip():
                report.skip(SkipReason.EMPTY)
                continue

            if _looks_generated(text, limits.max_line_length):
                report.skip(SkipReason.GENERATED)
                continue

            files.append(RepoFile(path=path, text=text))
            report.kept += 1
            report.total_bytes += len(raw)

            if report.kept > limits.max_files:
                raise RepositoryTooLarge(
                    f"Repository contains more than {limits.max_files} indexable files. "
                    "Try a smaller repository or a specific subdirectory fork."
                )
            if report.total_bytes > limits.max_total_text_bytes:
                raise RepositoryTooLarge(
                    "Repository text exceeds "
                    f"{limits.max_total_text_bytes // (1024 * 1024)} MB — refusing to process it."
                )

            if progress_callback and report.kept % 25 == 0:
                progress_callback(report.kept, None)

    files.sort(key=lambda f: f.path)   # deterministic ingestion order
    return files, report


def repo_files_to_documents(files: Iterable[RepoFile], repo: RepoRef, ref: str) -> list[Document]:
    """
    One Document per file, carrying the metadata citations and chunking need.

    ``source`` is the repository-relative path rather than a basename so that
    two ``__init__.py`` files in different packages stay distinct all the way
    through retrieval and citation. ``start_line`` is added later, during
    chunking, where the character offset within the file is known.
    """
    return [
        Document(
            page_content=f.text,
            metadata={
                "source": f.path,
                "source_type": SOURCE_TYPE_REPO,
                "repo": repo.slug,
                "ref": ref,
                "page": 0,
            },
        )
        for f in files
    ]


# ── Network layer ─────────────────────────────────────────────────────────────

def _headers(token: str | None) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "rag-doc-qa/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def resolve_repository(repo: RepoRef, token: str | None = None,
                       limits: RepoLimits = DEFAULT_LIMITS) -> tuple[str, int]:
    """
    Resolve the ref to use and check the repository's size before downloading.

    One API call. Worth it: the response carries ``size`` (KB), which lets an
    oversized repository be refused before any bytes are transferred, and
    ``default_branch``, which avoids guessing between main and master.
    """
    url = f"{GITHUB_API}/repos/{quote(repo.owner, safe='')}/{quote(repo.name, safe='')}"
    try:
        response = requests.get(url, headers=_headers(token), timeout=HTTP_TIMEOUT)
    except requests.RequestException as exc:
        raise RepositoryError(f"Could not reach GitHub: {exc}") from exc

    if response.status_code == 404:
        raise RepositoryNotFound(
            f"Repository {repo.slug} not found. It may be private, renamed, or misspelled."
        )
    if response.status_code == 403:
        raise RepositoryError(
            "GitHub rate limit reached (60 requests/hour without a token). "
            "Wait an hour or add a personal access token."
        )
    if not response.ok:
        raise RepositoryError(f"GitHub returned {response.status_code} for {repo.slug}.")

    payload = response.json()
    size_kb = int(payload.get("size") or 0)
    if size_kb > limits.max_repo_kb:
        raise RepositoryTooLarge(
            f"{repo.slug} is about {size_kb // 1024} MB; the limit is "
            f"{limits.max_repo_kb // 1024} MB."
        )

    ref = repo.ref or payload.get("default_branch") or "main"
    return ref, size_kb


def download_archive(repo: RepoRef, ref: str, token: str | None = None,
                     limits: RepoLimits = DEFAULT_LIMITS,
                     progress_callback=None) -> io.BytesIO:
    """
    Download the repository tarball in one request, streamed with a size cap.

    One archive request instead of walking the Contents API file by file,
    which would burn the 60 requests/hour unauthenticated limit on a single
    medium repository.
    """
    url = (
        f"{GITHUB_API}/repos/{quote(repo.owner, safe='')}/{quote(repo.name, safe='')}"
        f"/tarball/{quote(ref, safe='/')}"
    )
    try:
        response = requests.get(
            url, headers=_headers(token), timeout=HTTP_TIMEOUT, stream=True
        )
    except requests.RequestException as exc:
        raise RepositoryError(f"Could not download {repo.slug}: {exc}") from exc

    if response.status_code == 404:
        raise RepositoryNotFound(f"Branch or tag {ref!r} not found in {repo.slug}.")
    if not response.ok:
        raise RepositoryError(
            f"GitHub returned {response.status_code} downloading {repo.slug}@{ref}."
        )

    buffer = io.BytesIO()
    downloaded = 0
    for block in response.iter_content(chunk_size=DOWNLOAD_CHUNK):
        if not block:
            continue
        downloaded += len(block)
        if downloaded > limits.max_archive_bytes:
            response.close()
            raise RepositoryTooLarge(
                f"Download exceeded {limits.max_archive_bytes // (1024 * 1024)} MB."
            )
        buffer.write(block)
        if progress_callback:
            progress_callback(downloaded, None)

    buffer.seek(0)
    return buffer


def load_repository(url: str, token: str | None = None,
                    limits: RepoLimits = DEFAULT_LIMITS,
                    on_phase=None) -> tuple[list[Document], RepoRef, str, IngestReport]:
    """
    Full pipeline: URL -> Documents ready for ``chunk_documents``.

    on_phase: optional callable(phase, current, total) for UI progress, where
    phase is one of "resolve", "download", "read".
    """
    repo = parse_repo_url(url)

    def phase(name):
        return (lambda c, t: on_phase(name, c, t)) if on_phase else None

    if on_phase:
        on_phase("resolve", 0, None)
    ref, _size_kb = resolve_repository(repo, token=token, limits=limits)

    archive = download_archive(
        repo, ref, token=token, limits=limits, progress_callback=phase("download")
    )
    files, report = read_archive(archive, limits=limits, progress_callback=phase("read"))

    if not files:
        raise RepositoryEmpty(
            f"No indexable source or documentation files found in {repo.slug}@{ref}."
        )

    documents = repo_files_to_documents(files, repo, ref)
    print(f"[INFO] {repo.slug}@{ref}: kept {report.kept} file(s), "
          f"skipped {report.skipped_total}.")
    return documents, repo, ref, report
