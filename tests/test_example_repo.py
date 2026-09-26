"""
The bundled example repository, ingestion from a tarball on disk, and the
environment-driven repository limits.
"""

import hashlib
import io
import tarfile

import pytest

from src.example_repo import (
    EXAMPLE_ARCHIVE,
    EXAMPLE_ARCHIVE_SHA256,
    EXAMPLE_REPO_URL,
    load_example_repository,
)
from src.repo_ingestion import (
    RepoLimits,
    RepoRef,
    RepositoryEmpty,
    RepositoryError,
    limits_from_env,
    load_repository_archive,
    parse_repo_url,
)

MB = 1024 * 1024


def _tarball(entries: dict) -> bytes:
    """A GitHub-style .tar.gz: every path under one top-level folder."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in entries.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_bundled_archive_matches_its_pinned_sha256():
    with open(EXAMPLE_ARCHIVE, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == EXAMPLE_ARCHIVE_SHA256


def test_example_url_parses_to_the_pinned_tag():
    ref = parse_repo_url(EXAMPLE_REPO_URL)
    assert (ref.slug, ref.ref) == ("pallets/itsdangerous", "2.2.0")


def test_example_loads_from_disk_without_touching_the_network(monkeypatch):
    import requests

    def no_network(*_args, **_kwargs):
        raise AssertionError("the example must never call GitHub")

    monkeypatch.setattr(requests, "get", no_network)

    docs, repo, ref, report = load_example_repository()

    assert repo.label(ref) == "pallets/itsdangerous@2.2.0"
    assert report.kept == 48
    assert "src/itsdangerous/timed.py" in {d.metadata["source"] for d in docs}


def test_load_repository_archive_reads_a_local_tarball(tmp_path):
    archive = tmp_path / "repo.tar.gz"
    archive.write_bytes(_tarball({
        "owner-repo-abc123/README.md": "# Hello\n\nSome documentation.\n",
        "owner-repo-abc123/pkg/mod.py": "def answer():\n    return 42\n",
    }))

    docs, repo, ref, _report = load_repository_archive(str(archive), RepoRef("owner", "repo"), "v1")

    assert repo.label(ref) == "owner/repo@v1"
    assert {d.metadata["source"] for d in docs} == {"README.md", "pkg/mod.py"}


def test_load_repository_archive_refuses_a_checksum_mismatch(tmp_path):
    archive = tmp_path / "repo.tar.gz"
    archive.write_bytes(_tarball({"root/a.py": "x = 1\n"}))

    with pytest.raises(RepositoryError, match="SHA-256"):
        load_repository_archive(str(archive), RepoRef("o", "r"), "v1", sha256="0" * 64)


def test_load_repository_archive_with_nothing_indexable_raises(tmp_path):
    archive = tmp_path / "repo.tar.gz"
    archive.write_bytes(_tarball({"root/picture.png": "not really a png"}))

    with pytest.raises(RepositoryEmpty):
        load_repository_archive(str(archive), RepoRef("o", "r"), "v1")


def test_limits_from_env_keeps_the_defaults_when_unset():
    assert limits_from_env({}) == RepoLimits()
    assert limits_from_env({"RAG_REPO_MAX_FILES": " "}) == RepoLimits()


def test_limits_from_env_applies_the_demo_overrides():
    limits = limits_from_env({
        "RAG_REPO_MAX_FILES": "300",
        "RAG_REPO_MAX_ARCHIVE_MB": "20",
        "RAG_REPO_MAX_TEXT_MB": "20",
    })
    assert limits.max_files == 300
    assert limits.max_archive_bytes == 20 * MB
    assert limits.max_total_text_bytes == 20 * MB
    assert limits.max_file_bytes == RepoLimits().max_file_bytes       # untouched


@pytest.mark.parametrize("value", ["0", "-5", "1.5", "lots"])
def test_limits_from_env_rejects_malformed_values(value):
    with pytest.raises(ValueError, match="RAG_REPO_MAX_FILES"):
        limits_from_env({"RAG_REPO_MAX_FILES": value})
