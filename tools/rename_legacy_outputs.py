#!/usr/bin/env python3
"""One-shot migration: rewrite legacy scenario/mode names in outputs/.

Maps:
  flight00              -> flights_chi_nyc
  flight02              -> flights_ithaca_reston
  STUDENT_HARD          -> MAIN     (headphones scenario only)
  STUDENT               -> SOFT     (headphones scenario only)

For each JSON under the target run directory it:
  1. Rewrites meta.scenario, meta.mode, and meta.config.tag (+ embedded
     scenario / mode fields under meta.config and history.metadata).
  2. Computes the new filename (the legacy filename embeds scenario and
     mode segments) and renames the file on disk.

It also renames the per-scenario subfolders themselves
(outputs/.../flight00/ -> outputs/.../flights_chi_nyc/).

Idempotent: running twice is a no-op (already-renamed files are skipped).

Usage:
  python tools/rename_legacy_outputs.py outputs/paper__REPS40__...
  python tools/rename_legacy_outputs.py outputs/paper__REPS40__... --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

SCENARIO_RENAMES = {
    "flight00": "flights_chi_nyc",
    "flight02": "flights_ithaca_reston",
}

# Mode renames apply only when scenario == "headphones".
HEADPHONES_MODE_RENAMES = {
    "STUDENT_HARD": "MAIN",
    "STUDENT": "SOFT",
}


def _rename_scenario(value: Any) -> Any:
    if isinstance(value, str) and value in SCENARIO_RENAMES:
        return SCENARIO_RENAMES[value]
    return value


def _rename_mode_for_headphones(scenario: Any, mode: Any) -> Any:
    if scenario == "headphones" and isinstance(mode, str) and mode in HEADPHONES_MODE_RENAMES:
        return HEADPHONES_MODE_RENAMES[mode]
    return mode


def rewrite_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Rewrite scenario/mode fields in a result payload. Returns (new, changed)."""
    changed = False
    meta = payload.get("meta") or {}
    if not isinstance(meta, dict):
        return payload, False

    # Determine the "effective scenario" used to decide whether the mode is
    # the headphones one — check both pre- and post-rename names.
    old_scenario = meta.get("scenario")
    old_mode = meta.get("mode")

    new_scenario = _rename_scenario(old_scenario)
    new_mode = _rename_mode_for_headphones(old_scenario, old_mode)

    if new_scenario != old_scenario:
        meta["scenario"] = new_scenario
        changed = True
    if new_mode != old_mode:
        meta["mode"] = new_mode
        changed = True

    # Mirror into meta.config.{tag, scenario, mode} if present.
    cfg = meta.get("config")
    if isinstance(cfg, dict):
        if _rename_scenario(cfg.get("tag")) != cfg.get("tag"):
            cfg["tag"] = _rename_scenario(cfg.get("tag"))
            changed = True
        if _rename_scenario(cfg.get("scenario")) != cfg.get("scenario"):
            cfg["scenario"] = _rename_scenario(cfg.get("scenario"))
            changed = True
        if (
            cfg.get("mode") is not None
            and _rename_mode_for_headphones(new_scenario, cfg.get("mode")) != cfg.get("mode")
        ):
            cfg["mode"] = _rename_mode_for_headphones(new_scenario, cfg.get("mode"))
            changed = True

    return payload, changed


def new_filename(old_name: str, new_scenario: str, new_mode: str) -> str:
    """Filenames look like <scenario>__<algo>__<mode>__api...__llm...__...json
    Rewrites only the first segment (scenario) and the mode segment (3rd).
    """
    parts = old_name.split("__")
    if len(parts) < 3:
        return old_name
    if parts[0] in SCENARIO_RENAMES:
        parts[0] = SCENARIO_RENAMES[parts[0]]
    elif parts[0] == new_scenario:
        pass  # already renamed
    if new_scenario == "headphones" and parts[2] in HEADPHONES_MODE_RENAMES:
        parts[2] = HEADPHONES_MODE_RENAMES[parts[2]]
    return "__".join(parts)


def migrate_dir(run_dir: Path, dry_run: bool) -> Dict[str, int]:
    counts = {"json_rewritten": 0, "files_renamed": 0, "dirs_renamed": 0, "skipped": 0}

    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    # 1) Rewrite + rename JSON files in-place within their current folder.
    for json_path in sorted(run_dir.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ! could not parse {json_path}: {e}", file=sys.stderr)
            counts["skipped"] += 1
            continue

        new_payload, changed = rewrite_payload(payload)
        new_scenario = (new_payload.get("meta") or {}).get("scenario", "")
        new_mode = (new_payload.get("meta") or {}).get("mode", "")
        new_name = new_filename(json_path.name, new_scenario, new_mode)

        if changed or new_name != json_path.name:
            if not dry_run:
                json_path.write_text(
                    json.dumps(new_payload, indent=2, default=str), encoding="utf-8"
                )
            if changed:
                counts["json_rewritten"] += 1
            if new_name != json_path.name:
                target = json_path.with_name(new_name)
                if not dry_run:
                    json_path.rename(target)
                counts["files_renamed"] += 1

    # 2) Rename per-scenario subfolders.
    for old_name, new_name in SCENARIO_RENAMES.items():
        old_dir = run_dir / old_name
        new_dir = run_dir / new_name
        if old_dir.is_dir():
            if new_dir.exists():
                print(f"  ! cannot rename {old_dir} -> {new_dir} (target exists)", file=sys.stderr)
                counts["skipped"] += 1
                continue
            if not dry_run:
                old_dir.rename(new_dir)
            counts["dirs_renamed"] += 1

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="Path to a run directory containing per-scenario subfolders")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change but write nothing")
    args = ap.parse_args()

    counts = migrate_dir(Path(args.run_dir), args.dry_run)
    prefix = "[DRY-RUN] " if args.dry_run else ""
    print(f"{prefix}JSON payloads rewritten: {counts['json_rewritten']}")
    print(f"{prefix}Files renamed:           {counts['files_renamed']}")
    print(f"{prefix}Folders renamed:         {counts['dirs_renamed']}")
    if counts["skipped"]:
        print(f"{prefix}Skipped:                 {counts['skipped']}")


if __name__ == "__main__":
    main()
