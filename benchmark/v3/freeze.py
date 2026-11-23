"""
Freeze the reviewed v3 query set.

    python -m benchmark.v3.freeze --write     # once, after label review
    python -m benchmark.v3.freeze --verify    # what the harness runs first

Two outputs:

split.json     every query assigned to ``dev`` (70%) or ``holdout`` (30%),
               stratified by (domain, qtype) so each query type is represented
               in both halves. Each stratum is shuffled with its own
               string-seeded RNG, so the assignment depends only on the seed
               and the query ids - not on file order or the Python version.
manifest.json  SHA-256 of every frozen file. ``verify`` recomputes them and
               refuses if any differs; benchmark/v3/evaluate.py calls it before
               scoring, and a test calls it in CI.

Frozen files: the query set, the split, and gold.py (the span and
near-duplicate rules are part of what a label means). Changing any of them
requires an explicit, documented re-freeze (``--write --refreeze REASON``),
which is recorded in the manifest. Tuning decisions are made on ``dev`` only;
``holdout`` is for confirming a change once, at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

V3_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(V3_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SPLIT_PATH = os.path.join(V3_DIR, "split.json")
MANIFEST_PATH = os.path.join(V3_DIR, "manifest.json")
SEED = 20261123
HOLDOUT_SHARE = 0.30
FROZEN_FILES = ("benchmark/v3/queries.yaml", "benchmark/v3/split.json", "benchmark/v3/gold.py")


class FrozenFileChanged(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stratified_split(queries: list, seed: int = SEED, share: float = HOLDOUT_SHARE) -> dict:
    """{query id: 'dev' | 'holdout'}, stratified by (domain, qtype)."""
    strata: dict[tuple, list[str]] = defaultdict(list)
    for q in queries:
        strata[(q["domain"], q["qtype"])].append(q["id"])
    out: dict[str, str] = {}
    for (domain, qtype), ids in sorted(strata.items()):
        ids = sorted(ids)
        random.Random(f"{seed}:{domain}:{qtype}").shuffle(ids)
        n_holdout = int(len(ids) * share + 0.5)      # round half up, per stratum
        for i, qid in enumerate(ids):
            out[qid] = "holdout" if i < n_holdout else "dev"
    return out


def split_counts(queries: list, assignment: dict) -> dict:
    counts = Counter((q["domain"], q["qtype"], assignment[q["id"]]) for q in queries)
    return {f"{d}/{t}/{s}": n for (d, t, s), n in sorted(counts.items())}


def _manifest_digest(files: dict) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()


def write(refreeze_reason: str | None = None) -> dict:
    from benchmark.v3 import dataset as D

    previous = None
    if os.path.exists(MANIFEST_PATH):
        if not refreeze_reason:
            raise SystemExit("manifest.json exists: the set is already frozen. A re-freeze needs "
                             "--refreeze REASON and invalidates every result scored before it.")
        with open(MANIFEST_PATH, encoding="utf-8") as handle:
            previous = json.load(handle)

    queries = D.load()["queries"]
    assignment = stratified_split(queries)
    split = {"seed": SEED, "holdout_share": HOLDOUT_SHARE, "stratified_by": ["domain", "qtype"],
             "counts": split_counts(queries, assignment), "assignment": assignment}
    with open(SPLIT_PATH, "w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=1, sort_keys=False)
        handle.write("\n")

    files = {rel: sha256_file(os.path.join(PROJECT_ROOT, rel)) for rel in FROZEN_FILES}
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
        "manifest_sha256": _manifest_digest(files),
        "n_queries": len(queries),
        "n_dev": sum(1 for v in assignment.values() if v == "dev"),
        "n_holdout": sum(1 for v in assignment.values() if v == "holdout"),
        "history": (previous or {}).get("history", []) + (
            [{"refrozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "reason": refreeze_reason, "previous_manifest_sha256": previous["manifest_sha256"]}]
            if previous else []),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
        handle.write("\n")
    return manifest


def verify(manifest_path: str = MANIFEST_PATH, root: str = PROJECT_ROOT) -> dict:
    """Recompute every frozen file's hash; raise FrozenFileChanged on any mismatch."""
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    changed = []
    for rel, digest in manifest["files"].items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            changed.append(f"{rel} (missing)")
        elif sha256_file(path) != digest:
            changed.append(rel)
    if _manifest_digest(manifest["files"]) != manifest["manifest_sha256"]:
        changed.append("manifest.json (file list edited)")
    if changed:
        raise FrozenFileChanged(
            "refusing to score: frozen file(s) changed since the freeze: " + ", ".join(changed)
            + ". Revert them, or re-freeze explicitly with "
            "`python -m benchmark.v3.freeze --write --refreeze REASON`.")
    return manifest


def load_split_assignments(path: str = SPLIT_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["assignment"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--refreeze", metavar="REASON")
    args = parser.parse_args()
    if args.write:
        manifest = write(args.refreeze)
        print(json.dumps({k: v for k, v in manifest.items() if k != "history"}, indent=1))
        return 0
    manifest = verify()
    print(f"frozen set intact: {manifest['n_queries']} queries "
          f"({manifest['n_dev']} dev / {manifest['n_holdout']} holdout), "
          f"manifest {manifest['manifest_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
