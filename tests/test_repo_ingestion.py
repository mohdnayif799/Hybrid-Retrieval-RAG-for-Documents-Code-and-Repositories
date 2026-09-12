"""
Repository ingestion tests — entirely offline.

Archives are built in memory, so the filtering, safety and limit logic is
exercised without a single GitHub request. The network layer
(resolve_repository / download_archive) is deliberately not tested here; it is
thin, and testing it would mean either hitting GitHub during CI or asserting
against a mock of requests that proves little.
"""

import io
import tarfile
import time

import pytest

from src.repo_ingestion import (
    DEFAULT_LIMITS,
    InvalidRepositoryURL,
    RepoLimits,
    RepoRef,
    RepositoryTooLarge,
    SkipReason,
    parse_repo_url,
    read_archive,
    repo_files_to_documents,
    safe_member_path,
)

ROOT = "owner-repo-abc1234"


def make_tarball(entries: dict, root: str = ROOT) -> io.BytesIO:
    """Build a GitHub-shaped .tar.gz in memory. Keys are repo-relative paths."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, content in entries.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            info = tarfile.TarInfo(name=f"{root}/{path}")
            info.size = len(data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def paths(files):
    return [f.path for f in files]


# ── URL parsing ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,owner,name,ref", [
    ("https://github.com/psf/requests", "psf", "requests", None),
    ("https://github.com/psf/requests/", "psf", "requests", None),
    ("https://github.com/psf/requests.git", "psf", "requests", None),
    ("http://github.com/psf/requests", "psf", "requests", None),
    ("github.com/psf/requests", "psf", "requests", None),
    ("psf/requests", "psf", "requests", None),
    ("https://www.github.com/psf/requests", "psf", "requests", None),
    ("https://github.com/psf/requests/tree/main", "psf", "requests", "main"),
    ("https://github.com/a/b/tree/feature/nested/branch", "a", "b", "feature/nested/branch"),
    ("https://github.com/a/b/blob/dev", "a", "b", "dev"),
    ("https://github.com/My-Org/my_repo.v2", "My-Org", "my_repo.v2", None),
])
def test_parse_valid_urls(url, owner, name, ref):
    assert parse_repo_url(url) == RepoRef(owner=owner, name=name, ref=ref)


@pytest.mark.parametrize("url", [
    "",
    "   ",
    "https://gitlab.com/owner/repo",          # wrong host
    "https://evil.com/github.com/owner/repo",  # host spoofing attempt
    "https://github.com/onlyowner",           # missing repo
    "https://github.com/../../etc/passwd",
    "https://github.com/owner/repo/tree/../../../etc",
    "https://127.0.0.1/owner/repo",           # SSRF attempt
    "https://github.com.evil.com/owner/repo",
])
def test_parse_rejects_bad_urls(url):
    with pytest.raises(InvalidRepositoryURL):
        parse_repo_url(url)


def test_repo_label_is_stable():
    assert RepoRef("psf", "requests").label("main") == "psf/requests@main"


# ── Path safety ───────────────────────────────────────────────────────────────

def test_safe_member_path_strips_github_root():
    assert safe_member_path(f"{ROOT}/src/app.py") == "src/app.py"


@pytest.mark.parametrize("name", [
    f"{ROOT}/../../../etc/passwd",
    f"{ROOT}/../outside.py",
    f"{ROOT}/a/../../b.py",
    "/etc/passwd",
    "C:/Windows/System32/evil.dll",
    f"{ROOT}\\..\\..\\evil.py",       # Windows separators
    "",
    ROOT,                              # root itself, nothing after stripping
])
def test_safe_member_path_rejects_traversal(name):
    assert safe_member_path(name) is None


def test_path_traversal_entries_are_skipped_not_ingested():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in [f"{ROOT}/../../evil.py", "/abs/evil.py", f"{ROOT}/good.py"]:
            data = b"print('x')"
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)

    files, report = read_archive(buf)
    assert paths(files) == ["good.py"]
    assert report.skipped[SkipReason.UNSAFE_PATH] == 2


def test_symlinks_are_never_followed_or_read():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        link = tarfile.TarInfo(name=f"{ROOT}/secrets.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)

        data = b"x = 1"
        real = tarfile.TarInfo(name=f"{ROOT}/real.py")
        real.size = len(data)
        tar.addfile(real, io.BytesIO(data))
    buf.seek(0)

    files, _ = read_archive(buf)
    assert paths(files) == ["real.py"]


# ── Filtering ─────────────────────────────────────────────────────────────────

def test_keeps_source_and_docs_drops_junk():
    archive = make_tarball({
        "src/app.py":                  "def main(): pass",
        "src/ui.tsx":                  "export const App = () => null;",
        "README.md":                   "# Project",
        "node_modules/left-pad/i.js":  "module.exports = 1;",
        ".git/config":                 "[core]",
        "venv/lib/site.py":            "x = 1",
        "dist/bundle.js":              "console.log(1)",
        "__pycache__/app.cpython.pyc": "\x00\x01",
        "package-lock.json":           '{"lockfileVersion": 3}',
        "poetry.lock":                 "content-hash = 'x'",
        "logo.png":                    "\x89PNG",
        "app.min.js":                  "var a=1",
    })
    files, report = read_archive(archive)

    assert paths(files) == ["README.md", "src/app.py", "src/ui.tsx"]
    assert report.kept == 3
    assert report.skipped_total == 9


def test_bare_filenames_like_dockerfile_are_kept():
    files, _ = read_archive(make_tarball({
        "Dockerfile": "FROM python:3.10",
        "Makefile":   "all:\n\techo hi",
        "mystery":    "no extension and not a known name",
    }))
    assert paths(files) == ["Dockerfile", "Makefile"]


def test_binary_content_with_a_text_extension_is_skipped():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"\xff\xfe\x00\x01binary garbage"
        info = tarfile.TarInfo(name=f"{ROOT}/fake.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    files, report = read_archive(buf)
    assert files == []
    assert report.skipped[SkipReason.BINARY] == 1


def test_minified_content_is_skipped_even_without_a_min_extension():
    files, report = read_archive(make_tarball({
        "vendor_bundle.js": "var a=1;" * 500,     # one enormous line
        "normal.js":        "const a = 1;\nconst b = 2;\n",
    }))
    assert paths(files) == ["normal.js"]
    assert report.skipped[SkipReason.GENERATED] == 1


def test_empty_files_are_skipped():
    files, report = read_archive(make_tarball({"__init__.py": "", "real.py": "x = 1"}))
    assert paths(files) == ["real.py"]
    assert report.skipped[SkipReason.EMPTY] == 1


def test_ingestion_order_is_deterministic():
    entries = {f"src/mod_{i}.py": f"x = {i}" for i in range(20)}
    a, _ = read_archive(make_tarball(entries))
    b, _ = read_archive(make_tarball(entries))
    assert paths(a) == paths(b) == sorted(paths(a))


# ── Limits ────────────────────────────────────────────────────────────────────

def test_oversized_single_file_is_skipped_not_fatal():
    limits = RepoLimits(max_file_bytes=100)
    files, report = read_archive(
        make_tarball({"big.py": "x" * 500, "small.py": "y = 1"}), limits=limits
    )
    assert paths(files) == ["small.py"]
    assert report.skipped[SkipReason.TOO_LARGE] == 1


def test_too_many_files_raises_rather_than_truncating():
    # A silently truncated index answers "I don't have enough information" for
    # content the user believes was ingested. Failing loudly is the lesser evil.
    limits = RepoLimits(max_files=5)
    entries = {f"src/m{i}.py": f"x = {i}" for i in range(20)}
    with pytest.raises(RepositoryTooLarge, match="more than 5 indexable files"):
        read_archive(make_tarball(entries), limits=limits)


def test_declared_size_bomb_is_refused_before_reading():
    # Header claims a huge size; the guard must fire on the declared total
    # rather than after buffering the payload.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"a" * 1000
        info = tarfile.TarInfo(name=f"{ROOT}/bomb.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    with pytest.raises(RepositoryTooLarge, match="expands to more than"):
        read_archive(buf, limits=RepoLimits(max_uncompressed_bytes=100))


def test_total_text_cap_is_enforced():
    limits = RepoLimits(max_total_text_bytes=200)
    entries = {f"m{i}.py": "x" * 100 for i in range(10)}
    with pytest.raises(RepositoryTooLarge, match="exceeds"):
        read_archive(make_tarball(entries), limits=limits)


def test_default_limits_admit_an_ordinary_repository():
    entries = {f"src/pkg{i}/mod{j}.py": f"def f{j}():\n    return {j}\n"
               for i in range(10) for j in range(20)}
    files, report = read_archive(make_tarball(entries), limits=DEFAULT_LIMITS)
    assert report.kept == 200 and len(files) == 200


# ── Documents / metadata ──────────────────────────────────────────────────────

def test_documents_carry_repository_metadata():
    files, _ = read_archive(make_tarball({"src/app.py": "def main(): pass"}))
    docs = repo_files_to_documents(files, RepoRef("psf", "requests"), "main")

    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["source"] == "src/app.py"        # relative path, not a basename
    assert meta["source_type"] == "repo"
    assert meta["repo"] == "psf/requests"
    assert meta["ref"] == "main"
    assert meta["page"] == 0


def test_same_named_files_in_different_packages_stay_distinct():
    files, _ = read_archive(make_tarball({
        "src/a/__init__.py": "A = 1",
        "src/b/__init__.py": "B = 2",
    }))
    docs = repo_files_to_documents(files, RepoRef("o", "r"), "main")
    assert {d.metadata["source"] for d in docs} == {"src/a/__init__.py", "src/b/__init__.py"}


def test_excluded_content_does_not_consume_the_retained_text_budget():
    # A repository with large checked-in assets must still index its source:
    # the text cap applies to what is KEPT, not to what is walked past.
    entries = {"assets/blob.png": "x" * 40_000, "src/app.py": "def main(): pass"}
    files, _ = read_archive(
        make_tarball(entries),
        limits=RepoLimits(max_total_text_bytes=1000),
    )
    assert paths(files) == ["src/app.py"]


def test_walk_guard_is_far_looser_than_the_download_cap():
    # Regression guard on the constants themselves. GitHub reports PACKED
    # repository size, so a working tree expands well beyond the compressed
    # download. A walk cap close to the download cap refuses valid repos.
    assert DEFAULT_LIMITS.max_uncompressed_bytes >= DEFAULT_LIMITS.max_archive_bytes * 8
