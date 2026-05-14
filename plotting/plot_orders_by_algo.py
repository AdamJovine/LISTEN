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

from plotting_helpers import (  # noqa: E402
    load_algo, load_rerank_baselines, get_scenario_display_name, expand_baseline_variants,
)

# Columns plotted on the x-axis (in order). Each column is a distinct
# (scenario, primary_mode) pair — a scenario can appear multiple times if it
# has more than one preference-utterance mode worth plotting separately.
COLUMNS: List[Tuple[str, str, str, str]] = [
    # (column_id, scenario, primary_mode, display_name)
    ("flights_ithaca_reston", "flights_ithaca_reston", "Complicated",            "Flights Ithaca → Reston"),
    ("flights_chi_nyc",       "flights_chi_nyc",       "Complicated_structured", "Flights Chicago → NYC"),
    ("headphones__MAIN",      "headphones",            "MAIN",                   "Headphones-MAIN"),
    ("headphones__SOFT",      "headphones",            "SOFT",                   "Headphones-SOFT"),
    ("exam",                  "exam",                  "REGISTRAR",              "Exam Scheduling"),
]

# Canonical section-order numbering — matches the SECTION_ORDERS list in
# scripts/paper_recreate.sh so the index labels on the orders plot are stable.
SECTION_ORDER_INDEX: Dict[Tuple[str, ...], int] = {
    ("persona", "attributes", "priorities"):  1,
    ("persona", "priorities", "attributes"):  2,
    ("attributes", "persona", "priorities"):  3,
    ("attributes", "priorities", "persona"):  4,
    ("priorities", "persona", "attributes"):  5,
    ("priorities", "attributes", "persona"):  6,
}

# Display order for algo columns within each scenario.
ALGO_COLUMNS = ["tournament", "utility", "full_batch", "baseline_random", "baseline_zscore", "human_rerank"]
ALGO_DISPLAY = {
    "tournament":      "LISTEN-T",
    "utility":         "LISTEN-U",
    "full_batch":      "full_batch",
    "baseline_random": "baseline-random",
    "baseline_zscore": "baseline-zscore",
    "human_rerank":    "human rerank",
}
ALGO_COLOR = {
    "tournament":      "#1f77b4",
    "utility":         "#ff7f0e",
    "full_batch":      "#d62728",
    "baseline_random": "#2ca02c",
    "baseline_zscore": "#AA3377",
    "human_rerank":    "#9467bd",
}
ALGO_MARKER = {
    "tournament":      "^",
    "utility":         "s",
    "full_batch":      "P",
    "baseline_random": "X",
    "baseline_zscore": "D",
    "human_rerank":    "v",
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


_SCENARIOS_IN_COLUMNS = {scen for _c, scen, _m, _d in COLUMNS}


_ALGO_LABEL_MAP = {
    "BaselineRandom": "baseline_random",
    "BaselineZscore": "baseline_zscore",
}


def _algo_name_for(algo_obj, meta_algo: str) -> str:
    """Map an experiment object + its meta.algo to our internal algo key."""
    label = getattr(algo_obj, "algo_label", None)
    if label in _ALGO_LABEL_MAP:
        return _ALGO_LABEL_MAP[label]
    return meta_algo


def collect_by_order(
    data_dir: Path,
    api_model: str,
    tournament_batch_size: int | None,
    section_order_filter: Tuple[str, ...] | None = None,
) -> Dict[Tuple[str, str, Tuple[str, ...] | None], List[float]]:
    """Return {(column_id, algo_name, section_order_tuple_or_None): [nars...]}.

    Filters: api_model matches; tournament/full_batch runs at the given batch
    size only. `tournament`/`utility`/`full_batch` runs are mode-specific
    (must match the column's primary_mode). Baseline runs are mode-agnostic
    and contribute to every column for their scenario; each baseline JSON
    is expanded into a BaselineRandom and (when zscore_winner is non-null)
    BaselineZscore variant.
    """
    data: Dict[Tuple[str, str, Tuple[str, ...] | None], List[float]] = defaultdict(list)
    raw_pairs: List[Tuple[Path, dict, object]] = []
    for path in sorted(data_dir.glob("**/*.json")):
        if path.name in ("run_info.json", "manifest.json"):
            continue
        try:
            raw = json.loads(path.read_text())
        except Exception:
            continue
        meta = raw.get("meta", {})
        if meta.get("api_model") != api_model:
            continue
        if meta.get("scenario") not in _SCENARIOS_IN_COLUMNS:
            continue
        algo_kind = meta.get("algo")
        if algo_kind in ("tournament", "full_batch") and tournament_batch_size is not None:
            if meta.get("batch_size") != tournament_batch_size:
                continue
        try:
            algo_obj = load_algo(raw)
        except Exception:
            continue
        raw_pairs.append((path, meta, algo_obj))

    # Expand baseline -> BaselineRandom + BaselineZscore (deterministic, deduped).
    bl_pairs: List[Tuple[Path, object]] = [(p, a) for p, m, a in raw_pairs if m.get("algo") == "baseline"]
    bl_index: Dict[str, List[object]] = {}
    for _p, a_exp in expand_baseline_variants(bl_pairs):
        bl_index.setdefault(str(_p), []).append(a_exp)

    expanded: List[Tuple[dict, object]] = []
    for path, meta, algo_obj in raw_pairs:
        if meta.get("algo") == "baseline":
            variants = bl_index.get(str(path), [algo_obj])
        else:
            variants = [algo_obj]
        for v in variants:
            expanded.append((meta, v))

    so_filter_norm = tuple(s.lower() for s in section_order_filter) if section_order_filter else None
    for meta, algo_obj in expanded:
        scen = meta.get("scenario")
        mode = meta.get("mode")
        algo_name = _algo_name_for(algo_obj, meta.get("algo", ""))
        try:
            nar = algo_obj.get_nar()
        except Exception:
            continue
        if nar is None:
            continue
        cfg = meta.get("config") or {}
        so = cfg.get("section_order")
        so_key: Tuple[str, ...] | None = tuple(so) if isinstance(so, list) else None
        # Order filter applies only to prompt-aware algos. Baseline + rerank
        # have no section_order and pass through unconditionally.
        if so_filter_norm is not None and algo_name in ("tournament", "utility", "full_batch"):
            if so_key is None or tuple(s.lower() for s in so_key) != so_filter_norm:
                continue
        for col_id, c_scen, primary_mode, _disp in COLUMNS:
            if c_scen != scen:
                continue
            # NAR is scored against the run's mode-specific human_sol, so every
            # algo (including baseline variants) must match the column's
            # primary_mode — otherwise the same deterministic z-score winner
            # ends up averaged against multiple different gold rankings.
            if mode != primary_mode:
                continue
            data[(col_id, algo_name, so_key)].append(nar)
    return data


def collect_rerank() -> Dict[str, List[float]]:
    """Rerank baselines are tied to canonical (scenario, primary_mode); only
    columns whose primary_mode matches the rerank source receive data."""
    rerank = load_rerank_baselines()
    out: Dict[str, List[float]] = defaultdict(list)
    for _label, algo in rerank:
        scen = algo.get_scenario()
        mode = algo.get_mode()
        for col_id, c_scen, primary_mode, _disp in COLUMNS:
            if c_scen == scen and primary_mode == mode:
                n = algo.get_nar()
                if n is not None:
                    out[col_id].append(n)
    return out


def write_table(
    data: Dict[Tuple[str, str, Tuple[str, ...] | None], List[float]],
    rerank: Dict[str, List[float]],
    out_path: Path,
) -> None:
    """Emit one row per (column, algo, section_order) with mean / 2SE / n."""
    import csv
    column_labels = {c: d for c, _s, _m, d in COLUMNS}
    rows: List[List[str]] = []
    rows.append(["column", "algo", "section_order", "n", "mean_nar", "two_se"])
    column_ids = [c for c, _s, _m, _d in COLUMNS]
    for col_id in column_ids:
        for algo in ALGO_COLUMNS:
            if algo == "human_rerank":
                vals = rerank.get(col_id, [])
                if not vals:
                    continue
                mu, err = mean_and_2se(vals)
                rows.append([column_labels[col_id], "human_rerank", "", str(len(vals)),
                             f"{mu:.4f}", f"{err:.4f}"])
                continue
            seen_orders = sorted({so for (c, a, so) in data if c == col_id and a == algo},
                                 key=lambda t: ",".join(t) if t else "")
            for so in seen_orders:
                vals = data.get((col_id, algo, so), [])
                if not vals:
                    continue
                mu, err = mean_and_2se(vals)
                so_str = ",".join(so) if so else ""
                rows.append([column_labels[col_id], algo, so_str, str(len(vals)),
                             f"{mu:.4f}", f"{err:.4f}"])
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved table -> {out_path}")


def plot_one_api(
    data: Dict[Tuple[str, str, Tuple[str, ...] | None], List[float]],
    rerank: Dict[str, List[float]],
    api_model: str,
    out_path: Path,
) -> None:
    column_ids = [c for c, _s, _m, _d in COLUMNS]
    column_labels = {c: d for c, _s, _m, d in COLUMNS}

    # Which algo columns actually have any data? Drop empty ones.
    used_algos: List[str] = []
    for a in ALGO_COLUMNS:
        if a == "human_rerank":
            if any(rerank.get(c) for c in column_ids):
                used_algos.append(a)
            continue
        if any((c, a, so) in data and data[(c, a, so)] for (c, _aa, so) in data if _aa == a):
            used_algos.append(a)

    if not used_algos:
        print(f"[skip] no usable data for api_model={api_model}")
        return

    n_col = len(column_ids)
    n_algo = len(used_algos)
    x_centers = np.arange(n_col)
    algo_offsets = np.linspace(-0.35, 0.35, n_algo) if n_algo > 1 else np.array([0.0])

    fig, ax = plt.subplots(figsize=(max(10, 2 * n_col + 2), 7.0))

    legend_handles: Dict[str, Any] = {}
    sub_jitter_spread = 0.08

    for ai, algo in enumerate(used_algos):
        color = ALGO_COLOR[algo]
        marker = ALGO_MARKER[algo]
        x_algo = x_centers + algo_offsets[ai]
        for si, col_id in enumerate(column_ids):
            if algo == "human_rerank":
                vals = rerank.get(col_id, [])
                if not vals:
                    continue
                mu, err = mean_and_2se(vals)
                ax.errorbar(
                    [x_algo[si]], [mu], yerr=[err],
                    fmt=marker, color=color, markersize=8, capsize=3,
                    linewidth=0, elinewidth=1.2,
                    label=ALGO_DISPLAY[algo] if algo not in legend_handles else None,
                )
                legend_handles.setdefault(algo, True)
                continue

            # Gather all section_orders present for this (col_id, algo).
            # Sort by canonical SECTION_ORDER_INDEX so points lay out left->right
            # as orders 1..6.
            orders = sorted([so for (c, a, so) in data if c == col_id and a == algo and data[(c, a, so)]],
                            key=lambda t: SECTION_ORDER_INDEX.get(t, 99))
            if not orders:
                continue
            n_sub = len(orders)
            sub_offsets = np.linspace(-sub_jitter_spread, sub_jitter_spread, n_sub) if n_sub > 1 else np.array([0.0])
            for oi, so in enumerate(orders):
                vals = data[(col_id, algo, so)]
                mu, err = mean_and_2se(vals)
                xi = x_algo[si] + sub_offsets[oi]
                ax.errorbar(
                    [xi], [mu], yerr=[err],
                    fmt=marker, color=color, markersize=6, capsize=2.5,
                    linewidth=0, elinewidth=1.0, alpha=0.85,
                    label=ALGO_DISPLAY[algo] if algo not in legend_handles else None,
                )
                idx = SECTION_ORDER_INDEX.get(so) if so is not None else None
                if idx is not None:
                    ax.annotate(str(idx), (xi, mu + err),
                                textcoords="offset points", xytext=(0, 5),
                                ha="center", fontsize=9, fontweight="bold", color=color)
                legend_handles.setdefault(algo, True)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([column_labels[c] for c in column_ids])
    ax.set_ylabel("Normalized Average Rank (mean +/- 2 SE)")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="gray", alpha=0.5)
    ax.legend(fontsize=14, loc="center left", title=None)

    # When the plot uses section-order indices, append a footer explaining 1..6.
    needs_key = any(
        so is not None and SECTION_ORDER_INDEX.get(so) is not None
        for (_c, _a, so) in data
        if data[(_c, _a, so)]
    )
    if needs_key:
        key_lines = "    ".join(
            f"{idx}: {','.join(order)}"
            for order, idx in sorted(SECTION_ORDER_INDEX.items(), key=lambda kv: kv[1])
        )
        fig.text(0.5, 0.01, key_lines, ha="center", fontsize=8, color="#444")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
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
    ap.add_argument("--aggregate-orders", action="store_true",
                    help="Collapse section_order sub-points into one aggregate per algo "
                         "(produces <api>__nar__scenario__by_algo.png instead of _orders.png).")
    ap.add_argument("--section-order", type=str, default=None,
                    help="Comma-separated section order to filter prompt-aware algos to "
                         "(e.g. 'persona,attributes,priorities'). Baselines and human_rerank "
                         "are unaffected.")
    args = ap.parse_args()

    api_models = args.api_model or ["groq", "gemini"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rerank = collect_rerank()

    so_filter = None
    if args.section_order:
        so_filter = tuple(s.strip().lower() for s in args.section_order.split(",") if s.strip())

    for api in api_models:
        data = collect_by_order(
            args.data_dir, api_model=api,
            tournament_batch_size=args.batch_size,
            section_order_filter=so_filter,
        )
        if args.aggregate_orders:
            agg: Dict[Tuple[str, str, Tuple[str, ...] | None], List[float]] = defaultdict(list)
            for (col_id, algo, _so), vals in data.items():
                agg[(col_id, algo, None)].extend(vals)
            data = agg
            stem = f"{api}__nar__scenario__by_algo"
        else:
            stem = f"{api}__nar__scenario__by_algo_orders"
        plot_one_api(data, rerank, api, args.output_dir / f"{stem}.png")
        write_table(data, rerank, args.output_dir / f"{stem}.csv")


if __name__ == "__main__":
    main()
