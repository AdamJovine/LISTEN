#!/usr/bin/env python
"""Cross-scenario × algo plot with one sub-point per section_order.

For each api_model, draws a figure where each scenario has a row of algo
columns. Within each column, every section_order found in the data
contributes one point (mean +/- 2 SE), so a column for tournament will
typically have 6 sub-points (one per permutation of persona/attributes/
priorities) while algos with only the default order show a single point.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plotting"))

from plotting_helpers import load_algo, load_rerank_baselines, get_scenario_display_name  # noqa: E402

# scenario -> (canonical mode, config file)
SCENARIOS: Dict[str, Tuple[str, str]] = {
    "flights_ithaca_reston": ("Complicated", "flights_ithaca_reston.yml"),
    "flights_chi_nyc":       ("Complicated_structured", "flights_chi_nyc.yml"),
    "headphones":            ("MAIN", "headphones.yml"),
    "exam":                  ("REGISTRAR", "exam.yml"),
}

# Display order for algo columns within each scenario.
ALGO_COLUMNS = ["tournament", "utility", "full_batch", "baseline", "human_rerank"]
ALGO_DISPLAY = {
    "tournament": "LISTEN-T",
    "utility":    "LISTEN-U",
    "full_batch": "full_batch",
    "baseline":   "baseline",
    "human_rerank": "human rerank",
}
ALGO_COLOR = {
    "tournament":   "#1F77B4",
    "utility":      "#E45756",
    "full_batch":   "#9467BD",
    "baseline":     "#7F7F7F",
    "human_rerank": "#2CA02C",
}


def mean_and_2se(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(vals) / n
    if n == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, 2 * math.sqrt(var / n)


def collect_by_order(
    data_dir: Path,
    api_model: str,
    tournament_batch_size: int | None,
) -> Dict[Tuple[str, str, Tuple[str, ...] | None], List[float]]:
    """Return {(scenario, algo, section_order_tuple_or_None): [nars...]}.

    Filters: api_model matches, mode = canonical mode for that scenario,
    tournament/full_batch runs at the given batch size (others ignored).
    """
    data: Dict[Tuple[str, str, Tuple[str, ...] | None], List[float]] = defaultdict(list)
    for path in sorted(data_dir.glob("**/*.json")):
        if path.name in ("run_info.json", "manifest.json"):
            continue
        try:
            raw = json.loads(path.read_text())
        except Exception:
            continue
        meta = raw.get("meta", {})
        scen = meta.get("scenario")
        algo = meta.get("algo")
        mode = meta.get("mode")
        api  = meta.get("api_model")
        if api != api_model:
            continue
        if scen not in SCENARIOS:
            continue
        canonical_mode, _ = SCENARIOS[scen]
        if mode != canonical_mode:
            continue
        if algo in ("tournament", "full_batch") and tournament_batch_size is not None:
            if meta.get("batch_size") != tournament_batch_size:
                continue
        cfg = meta.get("config") or {}
        so = cfg.get("section_order")
        so_key: Tuple[str, ...] | None = tuple(so) if isinstance(so, list) else None
        try:
            algo_obj = load_algo(raw)
            nar = algo_obj.get_nar()
        except Exception:
            continue
        if nar is None:
            continue
        data[(scen, algo, so_key)].append(nar)
    return data


def collect_rerank(scenarios: List[str]) -> Dict[str, List[float]]:
    rerank = load_rerank_baselines()
    out: Dict[str, List[float]] = defaultdict(list)
    for _label, algo in rerank:
        s = algo.get_scenario()
        if s not in scenarios:
            continue
        n = algo.get_nar()
        if n is not None:
            out[s].append(n)
    return out


def plot_one_api(
    data: Dict[Tuple[str, str, Tuple[str, ...] | None], List[float]],
    rerank: Dict[str, List[float]],
    api_model: str,
    out_path: Path,
) -> None:
    scenarios = list(SCENARIOS.keys())

    # Which algo columns actually have any data? Drop empty ones.
    used_algos: List[str] = []
    for a in ALGO_COLUMNS:
        if a == "human_rerank":
            if any(rerank.get(s) for s in scenarios):
                used_algos.append(a)
            continue
        if any((s, a, so) in data and data[(s, a, so)] for (s, _aa, so) in data if _aa == a):
            used_algos.append(a)

    if not used_algos:
        print(f"[skip] no usable data for api_model={api_model}")
        return

    n_scen = len(scenarios)
    n_algo = len(used_algos)
    x_centers = np.arange(n_scen)
    algo_offsets = np.linspace(-0.35, 0.35, n_algo) if n_algo > 1 else np.array([0.0])

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_scen + 1), 5.5))

    legend_handles: Dict[str, Any] = {}
    sub_jitter_spread = 0.08

    for ai, algo in enumerate(used_algos):
        color = ALGO_COLOR[algo]
        x_algo = x_centers + algo_offsets[ai]
        for si, scen in enumerate(scenarios):
            if algo == "human_rerank":
                vals = rerank.get(scen, [])
                if not vals:
                    continue
                mu, err = mean_and_2se(vals)
                ax.errorbar(
                    [x_algo[si]], [mu], yerr=[err],
                    fmt="D", color=color, markersize=7, capsize=3,
                    linewidth=0, elinewidth=1.2,
                    label=ALGO_DISPLAY[algo] if algo not in legend_handles else None,
                )
                legend_handles.setdefault(algo, True)
                continue

            # Gather all section_orders present for this (scen, algo)
            orders = sorted([so for (s, a, so) in data if s == scen and a == algo and data[(s, a, so)]],
                            key=lambda t: ",".join(t) if t else "")
            if not orders:
                continue
            n_sub = len(orders)
            sub_offsets = np.linspace(-sub_jitter_spread, sub_jitter_spread, n_sub) if n_sub > 1 else np.array([0.0])
            for oi, so in enumerate(orders):
                vals = data[(scen, algo, so)]
                mu, err = mean_and_2se(vals)
                ax.errorbar(
                    [x_algo[si] + sub_offsets[oi]], [mu], yerr=[err],
                    fmt="o", color=color, markersize=5, capsize=2.5,
                    linewidth=0, elinewidth=1.0, alpha=0.85,
                    label=ALGO_DISPLAY[algo] if algo not in legend_handles else None,
                )
                legend_handles.setdefault(algo, True)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"{get_scenario_display_name(s)}\n{SCENARIOS[s][0]}" for s in scenarios])
    ax.set_ylabel("NAR (mean +/- 2 SE)")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="gray", alpha=0.5)
    ax.legend(fontsize=9, loc="upper left", title=None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Run directory containing per-scenario subfolders of JSON outputs.")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "plots")
    ap.add_argument("--api-model", action="append", default=None,
                    help="Repeatable. If omitted, both groq and gemini are plotted.")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="batch_size filter for tournament/full_batch runs (default 32).")
    args = ap.parse_args()

    api_models = args.api_model or ["groq", "gemini"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rerank = collect_rerank(list(SCENARIOS.keys()))

    for api in api_models:
        data = collect_by_order(args.data_dir, api_model=api, tournament_batch_size=args.batch_size)
        out = args.output_dir / f"{api}__nar__scenario__by_algo_orders.png"
        plot_one_api(data, rerank, api, out)


if __name__ == "__main__":
    main()
