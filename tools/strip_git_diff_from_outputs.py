#!/usr/bin/env python3
"""One-shot: remove the entire `meta.git` block from every JSON under a run directory.

Earlier versions of run_algorithm.py captured working-tree git metadata
(commit hash, branch, dirty flag, dirty file list, and up to 50 KB of
diff text) into each output payload under `meta.git`. That data can
contain in-progress / private dev content and is not needed downstream.

This script walks a run directory, drops the entire `git` key from
`meta` in every payload that has one, and writes the file back. Idempotent.

Usage:
  python tools/strip_git_diff_from_outputs.py outputs/paper__REPS40__...
  python tools/strip_git_diff_from_outputs.py outputs/paper__REPS40__... --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def strip(run_dir: Path, dry_run: bool) -> dict:
    counts = {"scanned": 0, "stripped": 0, "errors": 0}
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    for json_path in sorted(run_dir.rglob("*.json")):
        counts["scanned"] += 1
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ! could not parse {json_path}: {e}", file=sys.stderr)
            counts["errors"] += 1
            continue

        meta = payload.get("meta")
        if not isinstance(meta, dict) or "git" not in meta:
            continue
        del meta["git"]
        counts["stripped"] += 1
        if not dry_run:
            json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="Path to a run directory containing per-scenario subfolders")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change but write nothing")
    args = ap.parse_args()

    counts = strip(Path(args.run_dir), args.dry_run)
    prefix = "[DRY-RUN] " if args.dry_run else ""
    print(f"{prefix}Scanned:  {counts['scanned']} JSON files")
    print(f"{prefix}Stripped: {counts['stripped']} payloads (removed meta.git)")
    if counts["errors"]:
        print(f"{prefix}Errors:   {counts['errors']}")


if __name__ == "__main__":
    main()
