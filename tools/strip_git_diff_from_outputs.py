#!/usr/bin/env python3
"""One-shot: remove `meta.git.git_diff` from every JSON under a run directory.

Earlier versions of run_algorithm.py captured the working-tree dirty diff
(up to 50 KB) into each output payload as `meta.git.git_diff`. That field
can contain in-progress / private dev content; it also bypassed the
scenario rename done by tools/rename_legacy_outputs.py (the diff text is
opaque to that rewriter).

This script walks a run directory, drops the `git_diff` key from every
payload that has one, and writes the file back. Idempotent.

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

        git = (payload.get("meta") or {}).get("git")
        if not isinstance(git, dict) or "git_diff" not in git:
            continue
        del git["git_diff"]
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
    print(f"{prefix}Stripped: {counts['stripped']} payloads (removed meta.git.git_diff)")
    if counts["errors"]:
        print(f"{prefix}Errors:   {counts['errors']}")


if __name__ == "__main__":
    main()
