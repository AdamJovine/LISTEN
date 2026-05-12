#!/usr/bin/env python
"""Null out duplicate baseline zscore_winner entries in an output directory.

Zscore-avg is deterministic — every baseline run for the same (scenario, mode,
api_model) computes the same winner. Keeping all 40 reps surfaces 40 identical
BaselineZscore points in plots. This script keeps the zscore_winner in the
single oldest baseline JSON per (scenario, mode, api_model) group and zeros it
out in every other JSON in the group, so `expand_baseline_variants` emits
exactly one BaselineZscore entry per group while leaving the 40 random samples
untouched.

Idempotent: re-running does nothing once dedup has happened.

Usage:
  python scripts/dedupe_zscore.py <output_root>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def main(root: Path) -> int:
    if not root.is_dir():
        print(f"[ERR] not a directory: {root}", file=sys.stderr)
        return 1

    groups: dict[tuple, list[Path]] = defaultdict(list)
    for path in sorted(root.glob("**/*.json")):
        if path.name in ("run_info.json", "manifest.json"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        meta = data.get("meta", {})
        if meta.get("algo") != "baseline":
            continue
        key = (meta.get("scenario"), meta.get("mode"), meta.get("api_model"))
        groups[key].append(path)

    total_changed = 0
    kept_total = 0
    for key, paths in sorted(groups.items()):
        paths.sort(key=lambda p: p.stat().st_mtime)
        keeper = paths[0]
        kept_total += 1

        cleared = 0
        for p in paths[1:]:
            try:
                payload = json.loads(p.read_text())
            except Exception:
                continue
            history = (payload.get("results") or {}).get("history")
            if not isinstance(history, dict):
                continue
            batch_comps = history.get("batch_comparisons")
            if not isinstance(batch_comps, list):
                continue
            modified = False
            for entry in batch_comps:
                if isinstance(entry, dict) and entry.get("zscore_winner") is not None:
                    entry["zscore_winner"] = None
                    modified = True
            if modified:
                p.write_text(json.dumps(payload, indent=2, default=str))
                cleared += 1
                total_changed += 1

        print(f"  {key}: kept {keeper.name} | nulled zscore_winner in {cleared} of {len(paths)-1} others")

    print()
    print(f"[DONE] {kept_total} groups kept, {total_changed} files modified.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: dedupe_zscore.py <output_root>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
