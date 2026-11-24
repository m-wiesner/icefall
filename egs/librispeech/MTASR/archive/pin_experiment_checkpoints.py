#!/usr/bin/env python3
"""Hard-link MTASR checkpoints into a dated, rotation-safe local archive."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_revision(recipe: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=recipe,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root", type=Path, default=Path("exp"))
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    recipe = Path(__file__).resolve().parents[1]
    exp_root = (recipe / args.exp_root).resolve()
    archive_root = args.archive_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    records = []

    for source in sorted(exp_root.glob("**/checkpoint-*.pt")):
        if not source.is_file():
            continue
        relative = source.relative_to(exp_root)
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not os.path.samefile(source, destination):
                raise FileExistsError(
                    f"Refusing to replace different archive file: {destination}"
                )
        else:
            os.link(source, destination)
        stat = source.stat()
        records.append(
            {
                "source": str(source),
                "archive": str(destination),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "inode": stat.st_ino,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(recipe),
        "exp_root": str(exp_root),
        "archive_root": str(archive_root),
        "checkpoint_count": len(records),
        "logical_size_bytes": sum(item["size_bytes"] for item in records),
        "storage_note": "Archive entries are hard links and do not duplicate blocks.",
        "checkpoints": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Pinned {len(records)} checkpoints ({manifest['logical_size_bytes']} logical "
        f"bytes) under {archive_root}"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
