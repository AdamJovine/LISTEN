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

plt.rcParams["font.size"] *= 1.4
plt.rcParams["xtick.labelsize"] = 12  # x-tick labels: 1.2× matplotlib default of 10

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
    ("flights_ithaca_reston", "flights_ithaca_reston", "Complicated",            "Flights Ithaca\n→ Reston"),
    ("flights_chi_nyc",       "flights_chi_nyc",       "Complicated_structured", "Flights CHI->NYC"),
    ("headphones__MAIN",      "headphones",            "MAIN",                   "Headphones"),
    ("headphones__SOFT",      "headphones",            "SOFT",                   "Headphones-SOFT"),
    ("exam",                  "exam",                  "REGISTRAR",              "Exam Scheduling"),
]

# Canonical section-order numbering. The default order (persona, priorities,
# attributes) is index 1; remaining permutations fill 2..6 in their original
# relative order.
SECTION_ORDER_INDEX: Dict[Tuple[str, ...], int] = {
    ("persona", "priorities", "attributes"): 1,
    ("persona", "attributes", "priorities"): 2,
    ("attributes", "persona", "priorities"): 3,
    ("attributes", "priorities", "persona"): 4,
    ("priorities", "persona", "attributes"): 5,
    ("priorities", "attributes", "persona"): 6,
}

# Display order for algo columns within each scenario.
ALGO_COLUMNS = ["tournament", "utility", "full_batch", "baseline_random", "baseline_zscore", "human_rerank"]
ALGO_DISPLAY = {
    "tournament":      "LISTEN-T",
    "utility":         "LISTEN-U",
    "full_batch":      "baseline/full-batch",
    "baseline_random": "baseline/random",
    "baseline_zscore": "baseline/zscore-avg",
    "human_rerank":    "baseline/human-rerank",
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
BASELINE_ALGOS = {"full_batch", "baseline_random", "baseline_zscore", "human_rerank"}


def errorbar_style(algo: str, markersize: int = 11) -> Dict[str, Any]:
    """Errorbar style. Default markersize=11 suits the aggregated plot
    (one marker per algo per scenario). For the by-section-order variant
    (6 jittered markers per algo) pass markersize=7 to avoid overlap."""
    zorder = 5 if algo in BASELINE_ALGOS else 3
    return {
        "markersize": markersize,
        "capsize": 2.5,
        "elinewidth": 1.0,
        "markeredgewidth": 0.8,
        "markeredgecolor": "white",
        "alpha": 0.9,
        "zorder": zorder,
    }


def style_paper_axes(ax: Any, n_col: int) -> None:
    ax.set_ylim(-0.12, 0.78)
    ax.set_yticks(np.arange(-0.1, 0.71, 0.1))
    ax.set_xlim(-0.5, n_col - 0.5)
    for i in range(n_col):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#f1f3f7", alpha=0.55, zorder=0)
    for i in range(1, n_col):
        ax.axvline(i - 0.5, color="#bcbcbc", linestyle=":", linewidth=0.7, zorder=1)


def mean_and_2se(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(vals) / n
    if n == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, 2 * math.sqrt(var / n)


def annotate_section_order(ax: Any, x: float, y: float, err: float, idx: int, color: str) -> None:
    # Stagger odd indices above the upper error bar and even indices below the
    # lower one. Halves visual density at clusters where the 6 section_orders
    # land on nearly the same y (otherwise "123456" smushes into one blob).
    if idx % 2 == 1:
        anchor_y = y + err
        offset_y = 6
        va = "bottom"
    else:
        anchor_y = y - err
        offset_y = -6
        va = "top"
    ax.annotate(
        str(idx),
        (x, anchor_y),
        textcoords="offset points",
        xytext=(0, offset_y),
        ha="center",
        va=va,
        fontsize=9,
        fontweight="bold",
        color=color,
    )


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

    Filters: api_model matches; tournament runs at the given batch size only.
    Each baseline JSON is expanded into a BaselineRandom and
    (when zscore_winner is non-null) BaselineZscore variant.
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
        if algo_kind == "tournament" and tournament_batch_size is not None:
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
        # Order filter applies only to LISTEN-T and LISTEN-U. Baseline +
        # rerank pass through unconditionally because they have no
        # section_order. full_batch uses a single prompt layout, so its
        # data is included regardless of the active filter.
        if so_filter_norm is not None and algo_name in ("tournament", "utility"):
            if so_key is None or tuple(s.lower() for s in so_key) != so_filter_norm:
                continue
        for col_id, c_scen, primary_mode, _disp in COLUMNS:
            if c_scen != scen:
                continue
            # NAR is scored against the run's mode-specific human_sol, so every
            # algo, including baselines, must match the column's primary_mode.
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


def cap_raw_values(
    data: Dict[Tuple[str, str, Tuple[str, ...] | None], List[float]],
    reps_cap: int | None,
) -> Dict[Tuple[str, str, Tuple[str, ...] | None], List[float]]:
    if reps_cap is None:
        return data
    return {key: vals[:reps_cap] for key, vals in data.items()}


def cap_summary_counts(
    summary: Dict[Tuple[str, str, Tuple[str, ...] | None], Tuple[float, float, int]],
    reps_cap: int | None,
) -> Dict[Tuple[str, str, Tuple[str, ...] | None], Tuple[float, float, int]]:
    if reps_cap is None:
        return summary
    return {key: (mean, err, min(n, reps_cap)) for key, (mean, err, n) in summary.items()}


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
                                 key=lambda t: SECTION_ORDER_INDEX.get(t, 99) if t else 0)
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


def read_summary_table(
    path: Path,
) -> Dict[Tuple[str, str, Tuple[str, ...] | None], Tuple[float, float, int]]:
    """Read a previously emitted summary CSV.

    This is used when a PR/checkpoint includes plot CSVs but not the raw JSON
    runs. It can regenerate the PNG from the aggregated means and 2SE values.
    """
    import csv

    column_by_label = {display: col_id for col_id, _s, _m, display in COLUMNS}
    column_by_label["Flights Ithaca → Reston"] = "flights_ithaca_reston"
    column_by_label["Flights Chicago → NYC"] = "flights_chi_nyc"
    column_by_label["Headphones-MAIN"] = "headphones__MAIN"
    summary: Dict[Tuple[str, str, Tuple[str, ...] | None], Tuple[float, float, int]] = {}

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            col_label = row.get("column", "")
            col_id = column_by_label.get(col_label, col_label)
            algo = row.get("algo", "")
            so_raw = row.get("section_order", "").strip()
            so_key = tuple(part.strip() for part in so_raw.split(",") if part.strip()) if so_raw else None
            try:
                n = int(row.get("n", "0"))
                mean = float(row.get("mean_nar", "nan"))
                err = float(row.get("two_se", "nan"))
            except ValueError:
                continue
            if n <= 0:
                continue
            summary[(col_id, algo, so_key)] = (mean, err, n)

    return summary


def plot_one_api_summary(
    summary: Dict[Tuple[str, str, Tuple[str, ...] | None], Tuple[float, float, int]],
    api_model: str,
    out_path: Path,
) -> None:
    column_ids = [c for c, _s, _m, _d in COLUMNS]
    column_labels = {c: d for c, _s, _m, d in COLUMNS}

    used_algos: List[str] = []
    for algo in ALGO_COLUMNS:
        if any(c in column_ids and a == algo for (c, a, _so) in summary):
            used_algos.append(algo)

    if not used_algos:
        print(f"[skip] no summary rows for api_model={api_model}")
        return

    n_col = len(column_ids)
    n_algo = len(used_algos)
    x_centers = np.arange(n_col)
    algo_offsets = np.linspace(-0.4, 0.4, n_algo) if n_algo > 1 else np.array([0.0])

    fig, ax = plt.subplots(figsize=(max(10, 2 * n_col + 2), 7.0))
    legend_handles: Dict[str, Any] = {}
    sub_jitter_spread = 0.085

    for ai, algo in enumerate(used_algos):
        color = ALGO_COLOR[algo]
        marker = ALGO_MARKER[algo]
        x_algo = x_centers + algo_offsets[ai]
        for si, col_id in enumerate(column_ids):
            orders = sorted(
                [so for (c, a, so) in summary if c == col_id and a == algo],
                key=lambda t: SECTION_ORDER_INDEX.get(t, 99) if t else 0,
            )
            if not orders:
                continue
            sub_offsets = (
                np.linspace(-sub_jitter_spread, sub_jitter_spread, len(orders))
                if len(orders) > 1 else np.array([0.0])
            )
            for oi, so in enumerate(orders):
                mu, err, n = summary[(col_id, algo, so)]
                xi = x_algo[si] + sub_offsets[oi]
                ax.errorbar(
                    [xi], [mu], yerr=[err],
                    fmt=marker, color=color, linewidth=0,
                    label=ALGO_DISPLAY[algo] if algo not in legend_handles else None,
                    **errorbar_style(algo),
                )
                idx = SECTION_ORDER_INDEX.get(so) if so is not None else None
                if idx is not None:
                    annotate_section_order(ax, xi, mu, err, idx, color)
                legend_handles.setdefault(algo, True)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([column_labels[c] for c in column_ids], ha="center")
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("Normalized Average Rank (mean +/- 2 SE)", fontsize=14)
    style_paper_axes(ax, n_col)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="gray", alpha=0.5)
    # Horizontal legend above the axis. Wrap to 2 rows so 6 long algo labels
    # don't overflow the figure width.
    legend_ncol = min(n_algo, 3)
    ax.legend(fontsize=14, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=legend_ncol, frameon=False,
              handletextpad=0.35, columnspacing=1.0)

    present_orders = sorted(
        {
            so for (_c, _a, so) in summary
            if so is not None and SECTION_ORDER_INDEX.get(so) is not None
        },
        key=lambda t: SECTION_ORDER_INDEX[t],
    )
    if present_orders:
        ordered = [(so, SECTION_ORDER_INDEX[so]) for so in present_orders]
        half = (len(ordered) + 1) // 2
        line1 = "    ".join(f"{idx}: {','.join(order)}" for order, idx in ordered[:half])
        line2 = "    ".join(f"{idx}: {','.join(order)}" for order, idx in ordered[half:])
        # Section-order key at the bottom of the figure, below the x-tick row.
        fig.text(0.5, 0.04, line1, ha="center", va="bottom", fontsize=12, color="#444")
        fig.text(0.5, 0.01, line2, ha="center", va="bottom", fontsize=12, color="#444")
        # Reserve top for 2-row legend, bottom for x-ticks + 2-line key.
        fig.tight_layout(rect=(0, 0.10, 1.0, 0.88))
    else:
        # Reserve top for the 2-row horizontal legend sitting above the axis.
        fig.tight_layout(rect=(0, 0, 1.0, 0.88))

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_one_api(
    data: Dict[Tuple[str, str, Tuple[str, ...] | None], List[float]],
    rerank: Dict[str, List[float]],
    api_model: str,
    out_path: Path,
) -> None:
    column_ids = [c for c, _s, _m, _d in COLUMNS]
    column_labels = {c: d for c, _s, _m, d in COLUMNS}

    # full_batch belongs on the aggregated by-algo plot (one marker per algo),
    # but on the per-section-order plot it has no orders and would render as a
    # lone dot among the 6-variant clusters — so drop it there only.
    has_orders = any(so is not None for (_c, _a, so) in data)
    candidate_algos = [a for a in ALGO_COLUMNS if not (has_orders and a == "full_batch")]

    # Which algo columns actually have any data? Drop empty ones.
    used_algos: List[str] = []
    for a in candidate_algos:
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
    # Algos sit between -0.40 and +0.40 so even after sub-jitter the markers
    # stay strictly inside their scenario column (column boundary at +-0.5).
    algo_offsets = np.linspace(-0.40, 0.40, n_algo) if n_algo > 1 else np.array([0.0])

    # The by-section-order variant packs 6 jittered markers per algo and needs a
    # wider canvas; the aggregated variant (one marker per algo) would just look
    # sparse and shrink the fonts if rendered that wide.
    fig_w = max(13, 2.6 * n_col + 2) if has_orders else max(10, 2 * n_col + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 7.0))

    legend_handles: Dict[str, Any] = {}
    # sub_jitter_spread must:
    #   (a) leave a buffer between adjacent algo clusters
    #       (cluster width 2*s must fit inside the algo gap with margin), and
    #   (b) not push the outermost variant past the column edge at +-0.5.
    algo_gap = (0.8 / (n_algo - 1)) if n_algo > 1 else 1.0
    sub_jitter_spread = min(0.08, 0.35 * algo_gap, 0.5 - 0.40 - 0.02)

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
                    fmt=marker, color=color, linewidth=0,
                    label=ALGO_DISPLAY[algo] if algo not in legend_handles else None,
                    **errorbar_style(algo, markersize=8),
                )
                legend_handles.setdefault(algo, True)
                continue

            # Gather section_orders in canonical 1..6 order.
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
                    fmt=marker, color=color, linewidth=0,
                    label=ALGO_DISPLAY[algo] if algo not in legend_handles else None,
                    **errorbar_style(algo, markersize=8),
                )
                idx = SECTION_ORDER_INDEX.get(so) if so is not None else None
                if idx is not None:
                    annotate_section_order(ax, xi, mu, err, idx, color)
                legend_handles.setdefault(algo, True)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([column_labels[c] for c in column_ids], ha="center")
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("Normalized Average Rank (mean +/- 2 SE)", fontsize=14)
    style_paper_axes(ax, n_col)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="gray", alpha=0.5)
    # Horizontal legend above the axis. Wrap to 2 rows so 6 long algo labels
    # don't overflow the figure width.
    legend_ncol = min(n_algo, 3)
    ax.legend(fontsize=14, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=legend_ncol, frameon=False,
              handletextpad=0.35, columnspacing=1.0)

    present_orders = sorted(
        {
            so for (_c, _a, so) in data
            if data[(_c, _a, so)]
            and so is not None
            and SECTION_ORDER_INDEX.get(so) is not None
        },
        key=lambda t: SECTION_ORDER_INDEX[t],
    )
    if present_orders:
        ordered = [(so, SECTION_ORDER_INDEX[so]) for so in present_orders]
        half = (len(ordered) + 1) // 2
        line1 = "    ".join(f"{idx}: {','.join(order)}" for order, idx in ordered[:half])
        line2 = "    ".join(f"{idx}: {','.join(order)}" for order, idx in ordered[half:])
        # Section-order key at the bottom of the figure, below the x-tick row.
        fig.text(0.5, 0.04, line1, ha="center", va="bottom", fontsize=12, color="#444")
        fig.text(0.5, 0.01, line2, ha="center", va="bottom", fontsize=12, color="#444")
        # Reserve top for 2-row legend, bottom for x-ticks + 2-line key.
        fig.tight_layout(rect=(0, 0.10, 1.0, 0.88))
    else:
        # Reserve top for the 2-row horizontal legend sitting above the axis.
        fig.tight_layout(rect=(0, 0, 1.0, 0.88))

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
    ap.add_argument("--include-soft", action="store_true",
                    help="Include the Headphones-SOFT column in the aggregate main plot. "
                         "Non-aggregate order-sensitivity plots include it by default.")
    ap.add_argument("--section-order", type=str, default=None,
                    help="Comma-separated section order to filter prompt-aware algos to "
                         "(e.g. 'persona,attributes,priorities'). Baselines and human_rerank "
                         "are unaffected.")
    ap.add_argument("--reps-cap", type=int, default=40,
                    help="Maximum runs per plotted cell before aggregation (default 40; 0 disables).")
    args = ap.parse_args()

    global COLUMNS
    if args.aggregate_orders and not args.include_soft:
        COLUMNS = [col for col in COLUMNS if col[0] != "headphones__SOFT"]

    api_models = args.api_model or ["groq", "gemini"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rerank = collect_rerank()

    so_filter = None
    if args.section_order:
        so_filter = tuple(s.strip().lower() for s in args.section_order.split(",") if s.strip())
    reps_cap = args.reps_cap if args.reps_cap and args.reps_cap > 0 else None

    for api in api_models:
        data = collect_by_order(
            args.data_dir, api_model=api,
            tournament_batch_size=args.batch_size,
            section_order_filter=so_filter,
        )
        data = cap_raw_values(data, reps_cap)
        if args.aggregate_orders:
            agg: Dict[Tuple[str, str, Tuple[str, ...] | None], List[float]] = defaultdict(list)
            for (col_id, algo, _so), vals in data.items():
                agg[(col_id, algo, None)].extend(vals)
            data = cap_raw_values(agg, reps_cap)
            stem = f"{api}__nar__scenario__by_algo"
        else:
            stem = f"{api}__nar__scenario__by_algo_orders"
        csv_path = args.output_dir / f"{stem}.csv"
        has_raw_data = any(vals for vals in data.values())
        if not has_raw_data and csv_path.exists():
            print(f"[summary] no raw JSON rows for api_model={api}; regenerating from {csv_path}")
            summary = cap_summary_counts(read_summary_table(csv_path), reps_cap)
            plot_one_api_summary(
                summary, api, args.output_dir / f"{stem}.png",
            )
            continue
        plot_one_api(
            data, rerank, api, args.output_dir / f"{stem}.png",
        )
        write_table(data, rerank, csv_path)


if __name__ == "__main__":
    main()
