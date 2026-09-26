"""
The "Try an example" repository: pallets/itsdangerous at tag 2.2.0.

The tarball is committed under examples/ rather than fetched at runtime.
Cloud Run's outbound IPs are shared, so GitHub's 60-requests-per-hour
anonymous API limit is not ours alone, and any per-instance download cache is
lost on every scale-to-zero. A committed file makes the example work with no
network at all, in the container, in CI and locally.

The archive is GitHub's own tag tarball, byte for byte
(https://github.com/pallets/itsdangerous/archive/refs/tags/2.2.0.tar.gz).
itsdangerous is BSD-3-Clause licensed; its LICENSE.txt ships inside the
archive, which is what the licence requires for redistribution. The SHA-256
below is checked on every load, so a corrupted or swapped file is refused
rather than indexed.
"""

from __future__ import annotations

import os

from src.repo_ingestion import RepoRef, load_repository_archive

EXAMPLE_REPO = RepoRef(owner="pallets", name="itsdangerous", ref="2.2.0")
EXAMPLE_REPO_URL = f"https://github.com/{EXAMPLE_REPO.slug}/tree/{EXAMPLE_REPO.ref}"

EXAMPLE_ARCHIVE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "pallets-itsdangerous-2.2.0.tar.gz",
)
EXAMPLE_ARCHIVE_SHA256 = "7b0c6d4186e963b88489b69603b7ab2bf7c8e9eb4135a7b13b5f21bd4b937f2b"


def load_example_repository():
    """The example as ``load_repository()`` would return it, read from disk."""
    return load_repository_archive(
        EXAMPLE_ARCHIVE, EXAMPLE_REPO, EXAMPLE_REPO.ref, sha256=EXAMPLE_ARCHIVE_SHA256,
    )
