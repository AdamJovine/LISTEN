#!/usr/bin/env python
"""Keep one deterministic z-score baseline per output group.

The random baseline is sampled across repetitions, but the z-score winner is
deterministic for a given (scenario, mode, api_model). Keeping the z-score
winner in every baseline repetition overweights that deterministic point in
aggregate plots. This script keeps the oldest z-score winner per group and
sets the duplicate zscore_winner entries to null.

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
        if path.name in {"run_info.json", "manifest.json"}:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        meta = data.get("meta") or {}
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
        for path in paths[1:]:
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            history = (payload.get("results") or {}).get("history")
            if not isinstance(history, dict):
                continue
            batch_comparisons = history.get("batch_comparisons")
            if not isinstance(batch_comparisons, list):
                continue

            modified = False
            for entry in batch_comparisons:
                if isinstance(entry, dict) and entry.get("zscore_winner") is not None:
                    entry["zscore_winner"] = None
                    modified = True

            if modified:
                path.write_text(json.dumps(payload, indent=2, default=str))
                cleared += 1
                total_changed += 1

        print(
            f"  {key}: kept {keeper.name} | "
            f"nulled zscore_winner in {cleared} of {len(paths) - 1} others"
        )

    print()
    print(f"[DONE] {kept_total} groups kept, {total_changed} files modified.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: dedupe_zscore.py <output_root>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
