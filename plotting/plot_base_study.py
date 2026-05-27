#!/usr/bin/env python3
"""Preference-utterance ablation: BASE vs canonical-mode plot + NAR table.

Compares NAR between each scenario's canonical (with-preference) mode and
its BASE (no-preference) mode, with human-rerank baselines overlaid on the
plot. Every run is scored against the scenario's canonical human_sol so NAR
is comparable across modes.

Reads runs from --data-dir (which must contain per-scenario subfolders of
JSON outputs from run_algorithm.py).

Outputs (default outputs/plots/):
  base_study_nar__<api>.png
  base_vs_primary_table__<api>.csv
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

plt.rcParams["font.size"] *= 1.4
plt.rcParams["xtick.labelsize"] = 12  # x-tick labels: 1.2× matplotlib default of 10

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotting_helpers import load_algo, get_scenario_display_name

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
    ("headphones__MAIN",      "headphones",            "MAIN",                   "BASE", "Headphones"),
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
    "primary / tournament": ("#1f77b4", "o"),
    "base / tournament":    ("#1f77b4", "s"),
    "primary / utility":    ("#ff7f0e", "o"),
    "base / utility":       ("#ff7f0e", "s"),
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
    api_model: str | None = None,
    batch_size: int | None = None,
    section_order: List[str] | None = None,
    reps_cap: int | None = 40,
) -> Dict[Tuple[str, str, str], List[float]]:
    """Return {(col_id, algo, kind): [nar, ...]} where kind in {"primary", "base"}.

    A single JSON may match multiple columns when its (scenario, mode) pair
    appears in more than one column (e.g. headphones BASE matches both
    headphones__MAIN base and headphones__SOFT base). The run is scored
    independently against each column's primary-mode human_sol.

    Filters:
      api_model     — exact match on meta.api_model when set.
      batch_size    — applies only to tournament runs; utility ignores it
                      (tournament/utility have different canonical batch sizes).
      section_order — applies only to prompt-aware algos (tournament/utility)
                      when set.
      reps_cap      — per-cell cap on NAR count. JSONs are visited in sorted
                      path order so the first N are deterministic. Set to None
                      to disable. Defaults to 40 to enforce the REPS=40 budget
                      across cells when a directory accumulates re-runs at the
                      same (scenario, algo, mode, batch_size).
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
        if api_model is not None and meta.get("api_model") != api_model:
            continue
        if batch_size is not None and a == "tournament":
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
            key = (col_id, a, kind)
            if reps_cap is not None and len(data[key]) >= reps_cap:
                continue
            algo_obj.human_sol = list(HUMAN_SOL.get((scen, primary_mode), []))
            nar = algo_obj.get_nar()
            if nar is not None:
                data[key].append(nar)
    return data


# ── Plot + table ────────────────────────────────────────────────────────────

def write_plot(
    nars: Dict[Tuple[str, str, str], List[float]],
    out_path: Path,
) -> None:
    plot_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (col_id, algo, kind), vals in nars.items():
        plot_data[f"{kind} / {algo}"][col_id].extend(vals)

    column_ids = [col_id for col_id, _scen, _p, _b, _disp in COLUMNS]
    display_names = {col_id: disp for col_id, _scen, _p, _b, disp in COLUMNS}
    n_columns = len(column_ids)
    x = np.arange(n_columns)

    all_series_keys = [f"{kind} / {a}" for a in ALGOS for kind in MODE_KINDS]

    def _series_total(key: str) -> int:
        src = plot_data.get(key, {})
        return sum(len(src.get(c, [])) for c in column_ids)

    visible_keys = [k for k in all_series_keys if _series_total(k) > 0]
    skipped = [k for k in all_series_keys if k not in visible_keys]
    if skipped:
        print(f"[skip empty series] {skipped}")

    n_series = len(visible_keys)
    spread = 0.14
    offsets = np.linspace(-spread * (n_series - 1) / 2, spread * (n_series - 1) / 2, n_series)

    fig, ax = plt.subplots(figsize=(max(10, 2 * n_columns + 2), 7.0))

    for i in range(n_columns):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#f1f3f7", alpha=0.55, zorder=0)
    for i in range(1, n_columns):
        ax.axvline(i - 0.5, color="#bcbcbc", linestyle=":", linewidth=0.7, zorder=1)

    for i, key in enumerate(visible_keys):
        color, marker = SERIES_STYLE[key]
        src = plot_data.get(key, {})

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
            fmt=marker, color=color, markersize=10, capsize=3,
            markeredgewidth=1.0, markeredgecolor="white",
            label=label, linewidth=0, elinewidth=1.2, alpha=0.9, zorder=3,
        )

    x_labels = [display_names[col_id] for col_id in column_ids]
    ax.set_xticks(x)
    # 5 long scenario names at 14pt don't fit horizontally — rotate to avoid
    # the Flights Ithaca / Flights Chicago labels colliding.
    ax.set_xticklabels(x_labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("Normalized Average Rank (mean +/- 2 SE)", fontsize=14)
    ax.set_ylim(-0.12, 0.78)
    ax.set_yticks(np.arange(-0.1, 0.71, 0.1))
    ax.set_xlim(-0.5, n_columns - 0.5)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color="gray", alpha=0.5)
    # Horizontal legend above the axis so it never blocks data points.
    # For the typical 4-series layout (LISTEN-T/U × with-preference/no-pref)
    # use a tidy 2x2 grid instead of a lopsided 3+1 row.
    n_series = len(ax.get_legend_handles_labels()[0])
    if n_series == 4:
        legend_ncol = 2
    elif n_series > 0:
        legend_ncol = min(n_series, 3)
    else:
        legend_ncol = 1
    ax.legend(fontsize=14, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=legend_ncol, frameon=False,
              handletextpad=0.35, columnspacing=1.0)
    # Reserve top of the figure: 1 row if <=3 series, 2 rows otherwise (incl. 2x2).
    top_rect = 0.92 if n_series <= 3 else 0.86
    fig.tight_layout(rect=(0, 0, 1.0, top_rect))
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
        header += [
            f"{disp} primary NAR", f"{disp} primary n",
            f"{disp} BASE NAR", f"{disp} BASE n",
            f"{disp} primary_better",
        ]

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for col_id, scen, primary_mode, _base, display in COLUMNS:
            row: List[str] = [display, scen, primary_mode]
            for algo in ALGOS:
                prim_vals = nars.get((col_id, algo, "primary"), [])
                base_vals = nars.get((col_id, algo, "base"), [])
                prim = _mean(prim_vals)
                base = _mean(base_vals)
                if prim is None or base is None:
                    better = ""
                else:
                    better = "yes" if prim < base else "no"
                row += [_fmt(prim), str(len(prim_vals)), _fmt(base), str(len(base_vals)), better]
            writer.writerow(row)
    print(f"Saved table -> {out_path}")


def read_summary_table(path: Path) -> Dict[Tuple[str, str, str], List[float]]:
    """Read a previously emitted base-vs-primary CSV.

    Older bundled artifacts include only mean NAR columns, not per-cell 2SE.
    Returning repeated mean values preserves the point locations and visible
    sample-size labels while regenerating the PNG from the saved summary.
    """
    data: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    col_by_scenario_mode = {
        (scenario, primary_mode): col_id
        for col_id, scenario, primary_mode, _base_mode, _display in COLUMNS
    }

    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            col_id = col_by_scenario_mode.get((row.get("scenario", ""), row.get("primary_mode", "")))
            if not col_id:
                continue

            for algo in ALGOS:
                disp = ALGO_DISPLAY[algo]
                for kind, nar_column, n_column in (
                    ("primary", f"{disp} primary NAR", f"{disp} primary n"),
                    ("base", f"{disp} BASE NAR", f"{disp} BASE n"),
                ):
                    raw = (row.get(nar_column) or "").strip()
                    if not raw:
                        continue
                    try:
                        mean = float(raw)
                    except ValueError:
                        continue
                    n_raw = (row.get(n_column) or "").strip()
                    try:
                        n = int(n_raw) if n_raw else 40
                    except ValueError:
                        n = 40
                    data[(col_id, algo, kind)] = [mean] * n

    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Run directory containing per-scenario subfolders of JSON outputs.")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "plots",
                    help="Where to save the plot + CSV (default: outputs/plots/).")
    ap.add_argument("--api-model", action="append", default=None,
                    help="Repeatable. If omitted, both groq (llama) and gemini are plotted "
                         "as separate figures.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Tournament-only batch_size filter. Utility runs are not filtered "
                         "since they use a different canonical batch size.")
    ap.add_argument("--section-order", type=str, default=None,
                    help="Comma-separated section order to filter on (e.g. 'persona,attributes,priorities').")
    ap.add_argument("--reps", type=int, default=40,
                    help="Per-cell cap on NAR count (default 40). Set to 0 to disable.")
    args = ap.parse_args()

    section_order = None
    if args.section_order:
        section_order = [s.strip().lower() for s in args.section_order.split(",") if s.strip()]

    api_models = args.api_model or ["groq", "gemini"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reps_cap = args.reps if args.reps and args.reps > 0 else None
    for api in api_models:
        nars = collect_nars(
            args.data_dir,
            api_model=api,
            batch_size=args.batch_size,
            section_order=section_order,
            reps_cap=reps_cap,
        )
        has_listen = any(
            v for (col_id, algo, _kind), v in nars.items() if algo in ALGOS
        )
        if not has_listen:
            csv_path = args.output_dir / f"base_vs_primary_table__{api}.csv"
            if csv_path.exists():
                print(
                    f"[summary] no raw JSON rows for api_model={api}; "
                    f"regenerating from {csv_path}"
                )
                nars = read_summary_table(csv_path)
                has_listen = any(
                    v for (col_id, algo, _kind), v in nars.items() if algo in ALGOS
                )
            if not has_listen:
                print(f"[skip] no LISTEN runs found for api_model={api}")
                continue
        write_plot(nars, args.output_dir / f"base_study_nar__{api}.png")
        write_table(nars, args.output_dir / f"base_vs_primary_table__{api}.csv")


if __name__ == "__main__":
    main()
