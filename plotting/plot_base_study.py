#!/usr/bin/env python3
"""Preference-utterance ablation: BASE vs canonical-mode plot + NAR table.

Compares NAR between each scenario's canonical (with-preference) mode and
its BASE (no-preference) mode, with human-rerank baselines overlaid on the
plot. Every run is scored against the scenario's canonical human_sol so NAR
is comparable across modes.

Reads runs from --data-dir (which must contain per-scenario subfolders of
JSON outputs from run_algorithm.py).

Outputs (next to the PNG, default outputs/plots/):
  base_study_nar.png
  base_vs_primary_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotting_helpers import load_algo, load_rerank_baselines, get_scenario_display_name

# ── Config ──────────────────────────────────────────────────────────────────

ALGOS = ["tournament", "utility"]
ALGO_DISPLAY = {"tournament": "LISTEN-T", "utility": "LISTEN-U"}

# Columns plotted on the x-axis (in order). Each column is a distinct
# (scenario, primary_mode) pair compared against the same scenario's BASE
# mode. A scenario can appear more than once (e.g. headphones with MAIN vs
# headphones with SOFT) — each appears as its own column.
COLUMNS: List[Tuple[str, str, str, str, str]] = [
    # (column_id, scenario, primary_mode, base_mode, display_name)
    ("flights_ithaca_reston", "flights_ithaca_reston", "Complicated",            "BASE", "Flights Ithaca → Reston"),
    ("flights_chi_nyc",       "flights_chi_nyc",       "Complicated_structured", "BASE", "Flights Chicago → NYC"),
    ("headphones__MAIN",      "headphones",            "MAIN",                   "BASE", "Headphones-MAIN"),
    ("headphones__SOFT",      "headphones",            "SOFT",                   "BASE", "Headphones-SOFT"),
    ("exam",                  "exam",                  "REGISTRAR",              "BASE", "Exam Scheduling"),
]

# scenario -> config_file (loads per-mode human_sol)
SCENARIO_CFG: Dict[str, str] = {
    "flights_ithaca_reston": "flights_ithaca_reston.yml",
    "flights_chi_nyc":       "flights_chi_nyc.yml",
    "headphones":            "headphones.yml",
    "exam":                  "exam.yml",
}

MODE_KINDS = ["primary", "base"]
KIND_DISPLAY = {"primary": "with preference utterance", "base": "no-preference"}

SERIES_STYLE = {
    "primary / tournament": ("#1F77B4", "o"),
    "base / tournament":    ("#1F77B4", "s"),
    "primary / utility":    ("#E45756", "o"),
    "base / utility":       ("#E45756", "s"),
    "human rerank":         ("#2CA02C", "D"),
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_human_sols() -> Dict[Tuple[str, str], List[int]]:
    """Load human_sol for every (scenario, mode) pair referenced by COLUMNS."""
    import yaml
    cache: Dict[str, Dict] = {}
    out: Dict[Tuple[str, str], List[int]] = {}
    for _col_id, scen, primary, base, _disp in COLUMNS:
        if scen not in cache:
            with open(ROOT / "configs" / SCENARIO_CFG[scen], encoding="utf-8") as f:
                cache[scen] = yaml.safe_load(f) or {}
        modes_cfg = cache[scen].get("modes", {}) or {}
        for mode in (primary, base):
            key = (scen, mode)
            if key not in out:
                out[key] = (modes_cfg.get(mode, {}) or {}).get("human_sol", []) or []
    return out


HUMAN_SOL = load_human_sols()
SCENARIOS_IN_COLUMNS = {scen for _col_id, scen, _p, _b, _d in COLUMNS}


def mean_and_2se(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mu = sum(vals) / n
    if n == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, 2 * math.sqrt(var / n)


def collect_nars(
    data_dir: Path,
    batch_size: int | None = None,
    section_order: List[str] | None = None,
) -> Dict[Tuple[str, str, str], List[float]]:
    """Return {(col_id, algo, kind): [nar, ...]} where kind in {"primary", "base"}.

    A single JSON may match multiple columns when its (scenario, mode) pair
    appears in more than one column (e.g. headphones BASE matches both
    headphones__MAIN base and headphones__SOFT base). The run is scored
    independently against each column's primary-mode human_sol.
    """
    data: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    if not data_dir.is_dir():
        print(f"[ERR] data_dir does not exist: {data_dir}")
        return data
    so_norm = [str(x).lower() for x in section_order] if section_order else None
    for path in sorted(data_dir.glob("**/*.json")):
        if path.name in ("run_info.json", "manifest.json"):
            continue
        try:
            raw = json.loads(path.read_text())
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue
        meta = raw.get("meta", {})
        a = meta.get("algo", "")
        s = meta.get("scenario", "")
        m = meta.get("mode", "")
        if a not in ALGOS or s not in SCENARIOS_IN_COLUMNS:
            continue
        if batch_size is not None:
            bs = meta.get("batch_size")
            if bs is None or int(bs) != int(batch_size):
                continue
        if so_norm is not None:
            cfg = meta.get("config") or {}
            meta_so = cfg.get("section_order")
            if not isinstance(meta_so, list):
                continue
            if [str(x).lower() for x in meta_so] != so_norm:
                continue
        try:
            algo_obj = load_algo(raw)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue
        for col_id, scen, primary_mode, base_mode, _disp in COLUMNS:
            if scen != s:
                continue
            if m == primary_mode:
                kind = "primary"
            elif m == base_mode:
                kind = "base"
            else:
                continue
            algo_obj.human_sol = list(HUMAN_SOL.get((scen, primary_mode), []))
            nar = algo_obj.get_nar()
            if nar is not None:
                data[(col_id, a, kind)].append(nar)
    return data


def collect_human_rerank_nars() -> Dict[str, List[float]]:
    """Rerank baselines are tied to canonical (scenario, primary_mode) so they
    attach to the column whose primary_mode matches; non-canonical columns
    (e.g. headphones__SOFT) have no rerank data and are dropped automatically."""
    rerank_algos = load_rerank_baselines()
    out: Dict[str, List[float]] = defaultdict(list)
    for _label, algo in rerank_algos:
        scenario = algo.get_scenario()
        mode = algo.get_mode()
        for col_id, scen, primary_mode, _base, _disp in COLUMNS:
            if scen == scenario and primary_mode == mode:
                nar = algo.get_nar()
                if nar is not None:
                    out[col_id].append(nar)
    return out


# ── Plot + table ────────────────────────────────────────────────────────────

def write_plot(
    nars: Dict[Tuple[str, str, str], List[float]],
    out_path: Path,
) -> None:
    rerank_data = collect_human_rerank_nars()

    plot_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (col_id, algo, kind), vals in nars.items():
        plot_data[f"{kind} / {algo}"][col_id].extend(vals)

    column_ids = [col_id for col_id, _scen, _p, _b, _disp in COLUMNS]
    display_names = {col_id: disp for col_id, _scen, _p, _b, disp in COLUMNS}
    n_columns = len(column_ids)
    x = np.arange(n_columns)

    all_series_keys = [f"{kind} / {a}" for a in ALGOS for kind in MODE_KINDS] + ["human rerank"]

    def _series_total(key: str) -> int:
        src = rerank_data if key == "human rerank" else plot_data.get(key, {})
        return sum(len(src.get(c, [])) for c in column_ids)

    visible_keys = [k for k in all_series_keys if _series_total(k) > 0]
    skipped = [k for k in all_series_keys if k not in visible_keys]
    if skipped:
        print(f"[skip empty series] {skipped}")

    n_series = len(visible_keys)
    spread = 0.12
    offsets = np.linspace(-spread * (n_series - 1) / 2, spread * (n_series - 1) / 2, n_series)

    fig, ax = plt.subplots(figsize=(max(10, 2 * n_columns + 2), 5.5))

    for i, key in enumerate(visible_keys):
        color, marker = SERIES_STYLE[key]
        is_rerank = key == "human rerank"
        src = rerank_data if is_rerank else plot_data.get(key, {})

        means, errs, counts = [], [], []
        for col_id in column_ids:
            vals = src.get(col_id, [])
            if vals:
                mu, err = mean_and_2se(vals)
            else:
                mu, err = float("nan"), float("nan")
            means.append(mu)
            errs.append(err)
            counts.append(len(vals))
            print(f"  {key} / {col_id}: n={len(vals)}  mean={mu:.4f}  +/-2SE={err:.4f}")

        label = key
        for kind, display in KIND_DISPLAY.items():
            label = label.replace(kind, display, 1)
        label = label.replace("tournament", "LISTEN-T").replace("utility", "LISTEN-U")
        xs_i = x + offsets[i]
        ax.errorbar(
            xs_i, means, yerr=errs,
            fmt=marker, color=color, markersize=8, capsize=4,
            label=label, linewidth=0, elinewidth=1.5,
        )
        for xi, mi, ei, ci in zip(xs_i, means, errs, counts):
            if ci:
                ax.annotate(f"n={ci}", (xi, mi + ei),
                            textcoords="offset points", xytext=(0, 5),
                            ha="center", fontsize=6, color=color)

    x_labels = [display_names[col_id] for col_id in column_ids]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("NAR (mean +/- 2 SE)")
    ax.legend(fontsize=14, loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot -> {out_path}")


def write_table(
    nars: Dict[Tuple[str, str, str], List[float]],
    out_path: Path,
) -> None:
    def _mean(vals: List[float]) -> float | None:
        return (sum(vals) / len(vals)) if vals else None

    def _fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else ""

    header = ["column", "scenario", "primary_mode"]
    for algo in ALGOS:
        disp = ALGO_DISPLAY[algo]
        header += [f"{disp} primary NAR", f"{disp} BASE NAR", f"{disp} primary_better"]

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for col_id, scen, primary_mode, _base, display in COLUMNS:
            row: List[str] = [display, scen, primary_mode]
            for algo in ALGOS:
                prim = _mean(nars.get((col_id, algo, "primary"), []))
                base = _mean(nars.get((col_id, algo, "base"), []))
                if prim is None or base is None:
                    better = ""
                else:
                    better = "yes" if prim < base else "no"
                row += [_fmt(prim), _fmt(base), better]
            writer.writerow(row)
    print(f"Saved table -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Run directory containing per-scenario subfolders of JSON outputs.")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "plots",
                    help="Where to save the plot + CSV (default: outputs/plots/).")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="If set, only include runs at this batch size.")
    ap.add_argument("--section-order", type=str, default=None,
                    help="Comma-separated section order to filter on (e.g. 'persona,attributes,priorities').")
    args = ap.parse_args()

    section_order = None
    if args.section_order:
        section_order = [s.strip().lower() for s in args.section_order.split(",") if s.strip()]

    nars = collect_nars(args.data_dir, batch_size=args.batch_size, section_order=section_order)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plot(nars, args.output_dir / "base_study_nar.png")
    write_table(nars, args.output_dir / "base_vs_primary_table.csv")


if __name__ == "__main__":
    main()
