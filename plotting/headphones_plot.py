#!/usr/bin/env python3
"""
Plot normalized average rank (mean ± 2SE) of the final top-ranked item
for BOTH tournament and utility algorithms.

Supports two modes:
  1. Mode comparison (default): compare SOFT vs MAIN for the headphones scenario
  2. Layout comparison (--layout-dirs): compare prompt layouts (e.g. header_then_task_v1 vs task_then_header_v1)
"""

from __future__ import annotations

import argparse
import csv
from typing import Any, Dict, List, Tuple, TYPE_CHECKING
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] *= 1.4
plt.rcParams["xtick.labelsize"] = 12  # x-tick labels: 1.2× matplotlib default of 10

from plotting_helpers import (
    aggregate_algo_series,
    build_output_path,
    get_reps_cap,
    load_algo,
    scenario_outputs_root,
    compute_mean_and_stderr,
)

LISTEN_T_COLOR = "#1f77b4"
LISTEN_U_COLOR = "#ff7f0e"

if TYPE_CHECKING:
    from experiment import Experiment as AlgorithmClass


# ------------------------------------------------------------
# Metric
# ------------------------------------------------------------

def _compute_normalized_avg_rank(
    algo: "AlgorithmClass",
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """
    Compute normalized rank of the final top-ranked item.

    Returns a single-point series: x=[0], y=[normalized_rank]
    """

    history = algo._history or {}
    top_idx = None

    # Tournament algorithm: get last comparison's winner
    comps = history.get("batch_comparisons", [])
    if comps:
        last_comp = comps[-1]
        top_idx = last_comp.get("winner_idx")

    # Utility algorithm: get last iteration's winner
    if top_idx is None:
        iterations = history.get("iterations", [])
        if iterations:
            last_iter = iterations[-1]
            top_idx = last_iter.get("winner_idx")

    if top_idx is None:
        return [], [], {}

    rank = algo.get_rank(top_idx)

    if rank is None:
        return [], [], {}

    # Get number of items from solutions_df or metadata
    n_items = len(algo.solutions_df) if algo.solutions_df is not None else None
    if not n_items:
        metadata = history.get("metadata", {})
        n_items = metadata.get("n_solutions")

    if not n_items:
        return [], [], {}

    norm_rank = (rank - 1) / (n_items - 1)

    return [0], [norm_rank], {}


# ------------------------------------------------------------
# Mode comparison plot (original behavior)
# ------------------------------------------------------------

def headphones_plot(args: argparse.Namespace):
    # Default to figure_4 if no output_dir specified
    if args.output_dir is None:
        args.output_dir = "outputs/figure_3_try3"

    base_dir = scenario_outputs_root(args.scenario, args.output_dir)

    # If no mode specified, compare only the canonical headphones plot modes.
    if args.mode is None:
        if args.scenario == "headphones":
            modes = ["MAIN", "SOFT"]
        else:
            available_modes = set()
            for json_file in base_dir.glob("**/*.json"):
                try:
                    algo = load_algo(json_file)
                    # Only include modes that match the scenario
                    if algo.get_scenario() == args.scenario:
                        mode = algo.get_mode()
                        if mode:
                            available_modes.add(mode)
                except:
                    pass
            modes = sorted(available_modes)
    else:
        modes = [args.mode]

    if not modes:
        raise ValueError(f"No modes found for scenario '{args.scenario}' in {base_dir}")

    plt.figure(figsize=(8, 5))

    # Collect results for each mode and algorithm
    results = {}
    reps_cap = get_reps_cap(args)
    for mode in modes:
        results[mode] = {}
        for algo_name in ["tournament", "utility"]:
            args_copy = copy(args)
            args_copy.mode = mode
            args_copy.algo = algo_name
            # LISTEN-T: filter to batch size 32 (the only B with SOFT-mode
            # tournament data; MAIN has B in {2,4,8,16,32}). LISTEN-U has no
            # batch_size in its meta, so leave the filter open.
            if algo_name == "tournament":
                if not args_copy.batch_size:
                    args_copy.batch_size = 32
            else:
                args_copy.batch_size = None
            args_copy.reps_cap = reps_cap

            agg = aggregate_algo_series(args_copy, _compute_normalized_avg_rank, recursive=True)
            y = agg.mean[0]
            yerr = 2 * agg.se[0]
            results[mode][algo_name] = (y, yerr, agg.matched_runs)

    # Create grouped bar chart
    x = np.arange(len(modes))
    width = 0.35

    tournament_heights = [results[mode]["tournament"][0] for mode in modes]
    tournament_errs = [results[mode]["tournament"][1] for mode in modes]
    utility_heights = [results[mode]["utility"][0] for mode in modes]
    utility_errs = [results[mode]["utility"][1] for mode in modes]

    plt.bar(x - width/2, tournament_heights, width, yerr=tournament_errs,
            capsize=4, color=LISTEN_T_COLOR, label="LISTEN-T")
    plt.bar(x + width/2, utility_heights, width, yerr=utility_errs,
            capsize=4, color=LISTEN_U_COLOR, label="LISTEN-U")

    mode_display = {
        "MAIN": "Headphones",
        "SOFT": "Headphones-Soft",
    }
    plt.xticks(x, [mode_display.get(mode, f"{args.scenario.title()}-{mode.replace('_', ' ').title()}") for mode in modes], fontsize=16)
    plt.yticks(fontsize=14)
    plt.ylabel("Normalized Average Rank (mean +/- 2 SE)", fontsize=14)
    plt.ylim(0, max(tournament_heights + utility_heights) * 1.25)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=16)

    # Use args without algo/mode filter for output path; batch_size=32 always (LISTEN-T filter)
    args_no_algo = copy(args)
    args_no_algo.algo = None
    args_no_algo.mode = None if len(modes) > 1 else modes[0]
    args_no_algo.batch_size = args.batch_size or 32
    if getattr(args, "plot_dir", None):
        args_no_algo.output_dir = args.plot_dir
    out_path = build_output_path(args_no_algo, "norm-avg-rank-both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    csv_path = out_path.with_name(out_path.stem + "__n.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "algo", "n", "mean", "two_se"])
        for mode in modes:
            for algo_name in ("tournament", "utility"):
                y, yerr, n = results[mode][algo_name]
                writer.writerow([mode, algo_name, str(n), f"{y:.6f}", f"{yerr:.6f}"])
    print(f"Saved sample sizes -> {csv_path}")

    if args.show:
        plt.show()

    plt.close()
    return out_path


# ------------------------------------------------------------
# Layout comparison plot
# ------------------------------------------------------------

def layout_comparison_plot(args: argparse.Namespace):
    """Compare prompt layouts showing tournament + utility, across all modes found."""
    layout_dirs = args.layout_dirs
    layouts = [Path(d).name for d in layout_dirs]
    scenario = args.scenario

    from collections import defaultdict

    # Collect NARs: {layout: {mode: {algo: [nar, ...]}}}
    data: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    all_modes: set = set()
    reps_cap = get_reps_cap(args)

    for layout_dir in layout_dirs:
        layout_name = Path(layout_dir).name
        layout_path = Path(layout_dir)
        if not layout_path.is_absolute():
            from plotting_helpers import ROOT
            layout_path = ROOT / layout_path

        for json_file in sorted(layout_path.glob("*.json")):
            if json_file.name in ("run_info.json", "manifest.json", "experiment_spec.yml"):
                continue
            try:
                algo = load_algo(json_file)
            except Exception:
                continue

            if algo.get_scenario() != scenario:
                continue

            mode = algo.get_mode()
            if args.mode and mode != args.mode:
                continue

            algo_type = algo.get_algo().lower()
            if "tournament" in algo_type:
                algo_key = "tournament"
            elif "utility" in algo_type:
                algo_key = "utility"
            else:
                continue

            xs, ys, _ = _compute_normalized_avg_rank(algo)
            if ys:
                vals = data[layout_name][mode][algo_key]
                if reps_cap is None or len(vals) < reps_cap:
                    vals.append(ys[0])
                all_modes.add(mode)

    if not data:
        print(f"No data found for scenario={scenario}")
        return None

    modes = sorted(all_modes)
    algo_names = ["tournament", "utility"]

    import re
    def _clean_layout(name: str) -> str:
        # "header_then_task_v1" -> "header then task"
        name = name.replace("_", " ")
        return re.sub(r'\s+v\d+$', '', name)

    # One bar per (mode, algo, layout). Order: group all layouts of a given
    # (mode, algo) adjacent, then swap algo, then mode.
    bar_specs = []  # list of (mode, algo, layout)
    for mode in modes:
        for algo in algo_names:
            for layout in layouts:
                bar_specs.append((mode, algo, layout))

    # Insert separator gaps between (mode, algo) clusters
    cluster_size = len(layouts)
    inter_gap = 0.6  # gap between clusters
    bar_width = 0.8
    positions = []
    pos = 0.0
    for idx in range(len(bar_specs)):
        if idx > 0 and idx % cluster_size == 0:
            pos += inter_gap
        positions.append(pos)
        pos += bar_width

    fig, ax = plt.subplots(figsize=(max(10, len(bar_specs) * 0.9), 5))

    color_map = {"tournament": LISTEN_T_COLOR, "utility": LISTEN_U_COLOR}
    label_map = {"tournament": "LISTEN-T", "utility": "LISTEN-U"}
    legend_seen = set()

    bar_stats: List[Tuple[str, str, str, int, float, float]] = []
    for (mode, algo, layout), x_pos in zip(bar_specs, positions):
        vals = data.get(layout, {}).get(mode, {}).get(algo, [])
        if vals:
            m, s = compute_mean_and_stderr(vals)
        else:
            m, s = 0.0, 0.0
        color = color_map[algo]
        label = label_map[algo] if algo not in legend_seen else None
        legend_seen.add(algo)
        ax.bar(x_pos, m, bar_width, yerr=s, capsize=4,
               alpha=0.7, color=color, label=label)
        bar_stats.append((mode, algo, layout, len(vals), m, s))

    # X tick per bar
    x_labels = [_clean_layout(l) for (_m, _a, l) in bar_specs]
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=8.4, rotation=20, ha='right')

    # Cluster annotations below the bar labels (mode + algo)
    mode_display = {"SOFT": "Headphones-Soft", "MAIN": "Headphones"}
    for cluster_start in range(0, len(bar_specs), cluster_size):
        cluster = bar_specs[cluster_start:cluster_start + cluster_size]
        cluster_positions = positions[cluster_start:cluster_start + cluster_size]
        mode, algo, _ = cluster[0]
        center = sum(cluster_positions) / len(cluster_positions)
        ax.annotate(f"{label_map[algo]}\n{mode_display.get(mode, mode)}",
                    xy=(center, 0), xytext=(0, -42),
                    textcoords="offset points",
                    ha='center', va='top', fontsize=12.6, fontweight='bold',
                    annotation_clip=False)
    ax.set_ylabel("Normalized Average Rank (mean +/- 2SE)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=19.6)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    # Save to outputs/plots by default
    from plotting_helpers import ROOT as _ROOT
    plot_dir = Path(args.plot_dir) if args.plot_dir else _ROOT / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = f"layout_comparison__{scenario}.png"
    out_path = plot_dir / filename
    fig.savefig(out_path, dpi=150)

    csv_path = out_path.with_name(out_path.stem + "__n.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "algo", "layout", "n", "mean", "stderr"])
        for mode, algo, layout, n, mean, stderr in bar_stats:
            writer.writerow([mode, algo, layout, str(n), f"{mean:.6f}", f"{stderr:.6f}"])
    print(f"Saved sample sizes -> {csv_path}")

    if args.show:
        plt.show()
    plt.close()
    return out_path


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot normalized average rank for tournament and utility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare modes (SOFT vs MAIN):
  python headphones_plot.py --scenario headphones --output-dir outputs/some_dir

  # Compare prompt layouts:
  python headphones_plot.py --scenario headphones --mode MAIN \\
    --layout-dirs outputs/order_study_.../header_then_task_v1 outputs/order_study_.../task_then_header_v1
        """
    )

    ap.add_argument("--algo", choices=["tournament", "utility"], help="Algorithm filter (not used in mode comparison).")
    ap.add_argument("--scenario", required=True, help="Scenario name.")
    ap.add_argument("--mode", help="Scenario mode filter. If not specified, plots all modes found in data.")
    ap.add_argument("--iterations", type=int, help="Total iterations filter.")
    ap.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size filter.")
    ap.add_argument("--reps-cap", dest="reps_cap", type=int, default=40,
                    help="Maximum runs per plotted cell before aggregation (default: 40; 0 disables).")
    ap.add_argument("--api-model", help="API model filter.")
    ap.add_argument("--model-name", help="Model name filter.")
    ap.add_argument("--output-dir", dest="output_dir", help="Directory containing experiment JSON files.")
    ap.add_argument("--plot-dir", dest="plot_dir", help="Directory to save plots (default: same as --output-dir).")
    ap.add_argument("--show", action="store_true", help="Display plot interactively.")
    ap.add_argument(
        "--layout-dirs", dest="layout_dirs", nargs="+",
        help="Paths to layout subdirectories to compare (e.g. .../header_then_task_v1 .../task_then_header_v1). "
             "When provided, compares prompt layouts instead of modes."
    )

    args = ap.parse_args()

    if args.layout_dirs:
        out = layout_comparison_plot(args)
    else:
        out = headphones_plot(args)

    if out:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
