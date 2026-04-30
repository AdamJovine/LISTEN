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

# scenario -> (primary mode, BASE mode, config_file)
SCENARIOS: Dict[str, Tuple[str, str, str]] = {
    "flight02":   ("Complicated",            "BASE", "flight02.yml"),
    "flight00":   ("Complicated_structured", "BASE", "flight00.yml"),
    "headphones": ("STUDENT_HARD",           "BASE", "headphones.yml"),
    "exam":       ("REGISTRAR",              "BASE", "exam.yml"),
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

def load_canonical_human_sols() -> Dict[str, List[int]]:
    """Load each scenario's canonical (non-BASE) human_sol from configs/."""
    import yaml
    result: Dict[str, List[int]] = {}
    for scenario, (primary_mode, _base, cfg_file) in SCENARIOS.items():
        with open(ROOT / "configs" / cfg_file) as f:
            cfg = yaml.safe_load(f)
        result[scenario] = cfg.get("modes", {}).get(primary_mode, {}).get("human_sol", [])
    return result


CANONICAL_HUMAN_SOL = load_canonical_human_sols()


def mean_and_2se(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mu = sum(vals) / n
    if n == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, 2 * math.sqrt(var / n)


def collect_nars(data_dir: Path) -> Dict[Tuple[str, str, str], List[float]]:
    """Return {(scenario, algo, mode): [nar, ...]}.

    Every run (BASE or canonical) is scored against the scenario's canonical
    (non-BASE) human_sol so NAR is comparable across modes.
    """
    data: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
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
        if a not in ALGOS or s not in SCENARIOS:
            continue
        primary_mode, base_mode, _ = SCENARIOS[s]
        if m not in (primary_mode, base_mode):
            continue
        algo.human_sol = list(CANONICAL_HUMAN_SOL[s])
        nar = algo.get_nar()
        if nar is not None:
            data[(s, a, m)].append(nar)
    return data


def collect_human_rerank_nars() -> Dict[str, List[float]]:
    rerank_algos = load_rerank_baselines()
    data: Dict[str, List[float]] = defaultdict(list)
    for _label, algo in rerank_algos:
        scenario = algo.get_scenario()
        if scenario not in SCENARIOS:
            continue
        nar = algo.get_nar()
        if nar is not None:
            data[scenario].append(nar)
    return data


# ── Plot + table ────────────────────────────────────────────────────────────

def write_plot(
    nars: Dict[Tuple[str, str, str], List[float]],
    out_path: Path,
) -> None:
    rerank_data = collect_human_rerank_nars()

    plot_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (scenario, algo, mode), vals in nars.items():
        primary_mode, base_mode, _ = SCENARIOS[scenario]
        kind = "primary" if mode == primary_mode else ("base" if mode == base_mode else None)
        if kind is None:
            continue
        plot_data[f"{kind} / {algo}"][scenario].extend(vals)

    scenarios = list(SCENARIOS.keys())
    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)

    series_keys = [f"{kind} / {a}" for a in ALGOS for kind in MODE_KINDS] + ["human rerank"]
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

    x_labels = [f"{get_scenario_display_name(sc)}\nnon-Base Mode\n{SCENARIOS[sc][0]}"
                for sc in scenarios]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("NAR (mean +/- 2 SE)")
    ax.set_title("no-preference vs with preference utterance\n"
                 "circle = with preference utterance, square = no-preference")
    ax.legend(fontsize=8, loc="upper left")
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

    header = ["scenario", "primary_mode"]
    for algo in ALGOS:
        disp = ALGO_DISPLAY[algo]
        header += [f"{disp} primary NAR", f"{disp} BASE NAR", f"{disp} primary_better"]

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for scenario, (primary_mode, base_mode, _cfg) in SCENARIOS.items():
            row: List[str] = [get_scenario_display_name(scenario), primary_mode]
            for algo in ALGOS:
                prim = _mean(nars.get((scenario, algo, primary_mode), []))
                base = _mean(nars.get((scenario, algo, base_mode), []))
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
    args = ap.parse_args()

    nars = collect_nars(args.data_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_plot(nars, args.output_dir / "base_study_nar.png")
    write_table(nars, args.output_dir / "base_vs_primary_table.csv")


if __name__ == "__main__":
    main()
