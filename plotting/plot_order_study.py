#!/usr/bin/env python3
"""Order study: prompt-layout comparison plot + NAR table.

Compares NAR between header_then_task_v1 and task_then_header_v1 prompt
layouts across the canonical scenario/mode pairs, with human-rerank
baselines overlaid; also writes a CSV of mean NAR per (scenario, mode).

Reads runs from --data-dir (which must contain per-scenario subfolders of
JSON outputs from run_algorithm.py). Prompt format for each run is read from
`meta.config.comparison_prompt_variant_override` (tournament) or
`meta.config.utility_prompt_variant_override` (utility).

Outputs (next to the PNG, default outputs/plots/):
  order_study_nar.png
  order_study_nar_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotting_helpers import load_algo, load_rerank_baselines, get_scenario_display_name

# ── Config ──────────────────────────────────────────────────────────────────

LAYOUTS = ["header_then_task_v1", "task_then_header_v1"]
ALGOS = ["tournament", "utility"]

# Canonical scenario → mode pairs used for the plot.
PLOT_SCENARIOS: Dict[str, str] = {
    "flights_ithaca_reston": "Complicated",
    "flights_chi_nyc": "Complicated_structured",
    "headphones": "MAIN",
    "exam": "REGISTRAR",
}

# Rows for the CSV table — also includes headphones SOFT.
TABLE_ROWS: List[Tuple[str, str]] = [
    ("flights_ithaca_reston", "Complicated"),
    ("flights_chi_nyc",       "Complicated_structured"),
    ("headphones",            "SOFT"),
    ("headphones",            "MAIN"),
    ("exam",                  "REGISTRAR"),
]

TABLE_COLUMNS: List[Tuple[str, str, str]] = [
    ("header then task LISTEN-T", "header_then_task_v1", "tournament"),
    ("header then task LISTEN-U", "header_then_task_v1", "utility"),
    ("task then header LISTEN-T", "task_then_header_v1", "tournament"),
    ("task then header LISTEN-U", "task_then_header_v1", "utility"),
]

LAYOUT_LABELS = {
    "header_then_task_v1": "header then task",
    "task_then_header_v1": "task then header",
}

SERIES_STYLE = {
    "header_then_task_v1 / tournament": ("#1F77B4", "o"),
    "task_then_header_v1 / tournament": ("#1F77B4", "s"),
    "header_then_task_v1 / utility":    ("#E45756", "o"),
    "task_then_header_v1 / utility":    ("#E45756", "s"),
    "human rerank":                     ("#2CA02C", "D"),
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def mean_and_2se(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mu = sum(vals) / n
    if n == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, 2 * math.sqrt(var / n)


def _prompt_variant(meta: Dict[str, Any]) -> Optional[str]:
    cfg = meta.get("config") or {}
    val = cfg.get("comparison_prompt_variant_override") or cfg.get("utility_prompt_variant_override")
    return str(val) if val is not None else None


def collect_layout_nars(data_dir: Path) -> Dict[Tuple[str, str, str, str], List[float]]:
    """Return {(layout, algo, scenario, mode): [nar, ...]} keyed by prompt variant."""
    data: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    if not data_dir.is_dir():
        print(f"[ERR] data_dir does not exist: {data_dir}")
        return data
    for path in sorted(data_dir.glob("**/*.json")):
        if path.name in ("run_info.json", "manifest.json"):
            continue
        try:
            raw = json.loads(path.read_text())
            algo = load_algo(raw)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue
        meta = raw.get("meta", {})
        a = meta.get("algo", "")
        s = meta.get("scenario", "")
        m = meta.get("mode", "")
        layout = _prompt_variant(meta)
        if a not in ALGOS or layout not in LAYOUTS:
            continue
        nar = algo.get_nar()
        if nar is not None:
            data[(layout, a, s, m)].append(nar)
    return data


def collect_human_rerank_nars() -> Dict[str, List[float]]:
    rerank_algos = load_rerank_baselines()
    data: Dict[str, List[float]] = defaultdict(list)
    for _label, algo in rerank_algos:
        scenario = algo.get_scenario()
        if scenario not in PLOT_SCENARIOS:
            continue
        nar = algo.get_nar()
        if nar is not None:
            data[scenario].append(nar)
    return data


# ── Plot + table ────────────────────────────────────────────────────────────

def write_plot(
    layout_nars: Dict[Tuple[str, str, str, str], List[float]],
    out_path: Path,
) -> None:
    rerank_data = collect_human_rerank_nars()

    plot_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (layout, algo, scenario, mode), vals in layout_nars.items():
        if scenario not in PLOT_SCENARIOS:
            continue
        if mode != PLOT_SCENARIOS[scenario]:
            continue
        plot_data[f"{layout} / {algo}"][scenario].extend(vals)

    scenarios = list(PLOT_SCENARIOS.keys())
    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)

    series_keys = [f"{l} / {a}" for a in ALGOS for l in LAYOUTS] + ["human rerank"]
    n_series = len(series_keys)
    spread = 0.12
    offsets = np.linspace(-spread * (n_series - 1) / 2, spread * (n_series - 1) / 2, n_series)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, key in enumerate(series_keys):
        color, marker = SERIES_STYLE[key]
        is_rerank = key == "human rerank"
        src = rerank_data if is_rerank else plot_data.get(key, {})

        means, errs, counts = [], [], []
        for sc in scenarios:
            vals = src.get(sc, [])
            mu, err = mean_and_2se(vals)
            means.append(mu)
            errs.append(err)
            counts.append(len(vals))
            print(f"  {key} / {sc}: n={len(vals)}  mean={mu:.4f}  +/-2SE={err:.4f}")

        label = key
        for code, clean in LAYOUT_LABELS.items():
            label = label.replace(code, clean)
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

    x_labels = [get_scenario_display_name(sc) for sc in scenarios]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Normalized Average Rank (mean +/- 2 SE)")
    ax.legend(fontsize=14, loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot -> {out_path}")


def write_table(
    layout_nars: Dict[Tuple[str, str, str, str], List[float]],
    out_path: Path,
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "mode"] + [label for label, *_ in TABLE_COLUMNS])
        for scenario, mode in TABLE_ROWS:
            row: List[str] = [get_scenario_display_name(scenario), mode]
            for _label, layout, algo in TABLE_COLUMNS:
                vals = layout_nars.get((layout, algo, scenario, mode), [])
                cell = f"{(sum(vals) / len(vals)):.3f}" if vals else ""
                row.append(cell)
            writer.writerow(row)
    print(f"Saved table -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Run directory containing per-scenario subfolders of JSON outputs.")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "plots",
                    help="Where to save the plot + CSV (default: outputs/plots/).")
    args = ap.parse_args()

    layout_nars = collect_layout_nars(args.data_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plot(layout_nars, args.output_dir / "order_study_nar.png")
    write_table(layout_nars, args.output_dir / "order_study_nar_table.csv")


if __name__ == "__main__":
    main()
