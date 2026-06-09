#!/usr/bin/env python3
"""Shared helpers for LISTEN plotting scripts."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, TYPE_CHECKING

# Ensure project root is on sys.path for downstream imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment import Experiment

ComputeFn = Callable[[Dict[str, Any]], Tuple[List[int], List[float], Dict[str, Any]]]
AlgoComputeFn = Callable[["Experiment"], Tuple[List[int], List[float], Dict[str, Any]]]


@dataclass
class SeriesAggregate:
    xs: List[int]
    mean: List[float]
    se: List[float]
    matched_runs: int
    meta_sample: Dict[str, Any]
    extra: Dict[str, Any]


def compute_mean_and_stderr(vals: List[float]) -> Tuple[float, float]:
    """Compute mean and 2x standard error for a list of values."""
    import math

    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(vals) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    stderr = 2 * math.sqrt(var / n)  # 2x standard error for ~95% interval
    return (mean, stderr)


def get_reps_cap(args: argparse.Namespace, default: int | None = 40) -> int | None:
    cap = getattr(args, "reps_cap", default)
    if cap is None:
        return None
    cap = int(cap)
    return cap if cap > 0 else None


def add_common_plot_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Wire up shared CLI filters for plotting scripts."""
    ap.add_argument("--algo", default="tournament", choices=["tournament", "utility", "full_batch", "baseline"], help="Algorithm filter.")
    ap.add_argument("--scenario", required=True, help="Scenario name.")
    ap.add_argument("--mode", help="Scenario mode filter.")
    ap.add_argument("--iterations", type=int, help="Total iterations filter.")
    ap.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size filter.")
    ap.add_argument("--api-model", help="API model filter.")
    ap.add_argument("--model-name", help="Model name filter.")
    ap.add_argument("--output-dir", dest="output_dir", help="Custom input directory to read data from (default: outputs/<scenario>).")
    ap.add_argument("--save-dir", dest="save_dir", help="Directory to save plots (default: outputs/plots).")
    ap.add_argument("--output", dest="output", help="Explicit output file path for the plot (overrides --save-dir).")
    ap.add_argument("--reps-cap", dest="reps_cap", type=int, default=40,
                    help="Maximum runs per plotted cell before aggregation (default: 40; 0 disables).")
    ap.add_argument("--show", action="store_true", help="Display plot interactively.")
    return ap


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_algo(source: "str | Path | Dict[str, Any]") -> "Experiment":
    """Load an algorithm from a JSON file path or dict using Experiment.from_json."""
    # Load the raw JSON to get meta.batch_size
    if isinstance(source, (str, Path)):
        raw_data = load_payload(Path(source))
    else:
        raw_data = source

    algo = Experiment.from_json(source)

    # Copy batch_size from meta if not already in algo_config
    meta = raw_data.get("meta") or {}
    if "batch_size" not in algo.algo_config and meta.get("batch_size"):
        algo.algo_config["batch_size"] = meta["batch_size"]

    # Copy human_sol from meta.config.modes[mode] if not already set properly
    mode = meta.get("mode")
    modes = (meta.get("config") or {}).get("modes") or {}
    if mode and mode in modes:
        hs = modes[mode].get("human_sol")
        if hs:
            algo.human_sol = list(hs)

    return algo


def matches_algo(algo: "Experiment", args: argparse.Namespace) -> bool:
    """Check if an algorithm matches the CLI filter arguments using getters."""
    algo_name = (algo.get_algo() or "").lower()
    algo_arg = (args.algo or "").lower()
    if algo_arg and algo_name and algo_name != algo_arg and not algo_name.startswith(algo_arg):
        return False
    if algo.get_scenario() != args.scenario:
        return False
    if args.mode and algo.get_mode() != args.mode:
        return False
    if args.api_model and algo.algo_config.get("api_model") != args.api_model:
        return False
    if args.model_name and algo.algo_config.get("model_name") != args.model_name:
        return False
    if args.iterations and algo.iterations and int(algo.iterations) != int(args.iterations):
        return False
    batch_size = getattr(algo, "batch_size", None)
    if args.batch_size and batch_size and int(batch_size) != int(args.batch_size):
        return False
    return True


def matches_meta(meta: Dict[str, Any], args: argparse.Namespace) -> bool:
    """Check if metadata dict matches CLI filter arguments (legacy compatibility)."""
    algo_meta = (meta.get("algo") or "").lower()
    algo_arg = (args.algo or "").lower()
    if algo_arg and algo_meta and algo_meta != algo_arg and not algo_meta.startswith(algo_arg):
        return False
    if meta.get("scenario") != args.scenario:
        return False
    if args.mode and meta.get("mode") != args.mode:
        return False
    if args.api_model and meta.get("api_model") != args.api_model:
        return False
    if args.model_name and meta.get("model_name") != args.model_name:
        return False
    if args.iterations and meta.get("max_iters") and int(meta.get("max_iters")) != int(args.iterations):
        return False
    if args.batch_size and meta.get("batch_size") and int(meta.get("batch_size")) != int(args.batch_size):
        return False
    return True


def scenario_outputs_root(scenario: str, output_dir: str | None = None) -> Path:
    """Get the output directory for a scenario, or use custom output_dir if provided."""
    if output_dir:
        out_root = Path(output_dir)
    else:
        out_root = ROOT / "outputs" / scenario
    if not out_root.exists():
        raise FileNotFoundError(f"outputs directory not found at {out_root}")
    return out_root


def iter_matching_payloads(
    args: argparse.Namespace, out_root: Path | None = None
) -> Iterable[Tuple[Path, Dict[str, Any], Dict[str, Any]]]:
    """Iterate over matching payloads (legacy compatibility)."""
    output_dir = getattr(args, "output_dir", None)
    base = out_root or scenario_outputs_root(args.scenario, output_dir)
    for path in sorted(base.glob("**/*.json")):
        payload = load_payload(path)
        meta = payload.get("meta") or {}
        if matches_meta(meta, args):
            yield path, payload, meta


def iter_matching_algos(
    args: argparse.Namespace, out_root: Path | None = None, recursive: bool = False
) -> Iterable[Tuple[Path, "Experiment"]]:
    """Iterate over matching algorithm objects using Experiment.from_json."""
    output_dir = getattr(args, "output_dir", None)
    base = out_root or scenario_outputs_root(args.scenario, output_dir)
    pattern = "**/*.json" if recursive else "*.json"
    for path in sorted(base.glob(pattern)):
        try:
            algo = load_algo(path)
            if matches_algo(algo, args):
                yield path, algo
        except Exception:
            # Skip files that fail to load
            continue


def aggregate_iteration_series(
    args: argparse.Namespace, compute_fn: ComputeFn, out_root: Path | None = None
) -> SeriesAggregate:
    """Aggregate series data across matching payloads (legacy compatibility)."""
    series: Dict[int, List[float]] = defaultdict(list)
    matched = 0
    meta_sample: Dict[str, Any] = {}
    extra_info: Dict[str, Any] = {}
    reps_cap = get_reps_cap(args)

    for _, payload, meta in iter_matching_payloads(args, out_root):
        if reps_cap is not None and matched >= reps_cap:
            break
        xs, ys, info = compute_fn(payload)
        if not xs or not ys:
            continue
        matched += 1
        meta_sample = meta
        extra_info = info
        for x, y in zip(xs, ys):
            series[int(x)].append(float(y))

    if matched == 0:
        output_dir = getattr(args, "output_dir", None)
        base = out_root or scenario_outputs_root(args.scenario, output_dir)
        raise RuntimeError(f"No runs found matching filters under {base}")

    xs_sorted = sorted(series.keys())
    mean = []
    se = []
    for x in xs_sorted:
        m, s = compute_mean_and_stderr(series[x])
        mean.append(m)
        se.append(s)

    return SeriesAggregate(xs_sorted, mean, se, matched, meta_sample, extra_info)


def aggregate_algo_series(
    args: argparse.Namespace, compute_fn: AlgoComputeFn, out_root: Path | None = None, recursive: bool = False
) -> SeriesAggregate:
    """Aggregate series data across matching algorithm objects using Experiment.from_json."""
    series: Dict[int, List[float]] = defaultdict(list)
    matched = 0
    algo_sample: "Experiment | None" = None
    extra_info: Dict[str, Any] = {}
    reps_cap = get_reps_cap(args)

    for _, algo in iter_matching_algos(args, out_root, recursive=recursive):
        if reps_cap is not None and matched >= reps_cap:
            break
        xs, ys, info = compute_fn(algo)
        if not xs or not ys:
            continue
        matched += 1
        algo_sample = algo
        extra_info = info
        for x, y in zip(xs, ys):
            series[int(x)].append(float(y))

    if matched == 0:
        output_dir = getattr(args, "output_dir", None)
        base = out_root or scenario_outputs_root(args.scenario, output_dir)
        raise RuntimeError(f"No runs found matching filters under {base}")

    # Build meta_sample from algo_sample for compatibility
    meta_sample: Dict[str, Any] = {}
    if algo_sample:
        meta_sample = {
            "scenario": algo_sample.get_scenario(),
            "mode": algo_sample.get_mode(),
            "algo": algo_sample.get_algo(),
            "api_model": algo_sample.algo_config.get("api_model"),
            "model_name": algo_sample.algo_config.get("model_name"),
            "batch_size": getattr(algo_sample, "batch_size", None),
        }

    xs_sorted = sorted(series.keys())
    mean = []
    se = []
    for x in xs_sorted:
        m, s = compute_mean_and_stderr(series[x])
        mean.append(m)
        se.append(s)

    return SeriesAggregate(xs_sorted, mean, se, matched, meta_sample, extra_info)


def build_plot_title(args: argparse.Namespace, meta_sample: Dict[str, Any]) -> str:
    parts = [
        args.scenario,
        args.mode or meta_sample.get("mode"),
        args.algo,
        args.api_model or meta_sample.get("api_model"),
        getattr(args, "model_name", None) or meta_sample.get("model_name"),
    ]
    batch = args.batch_size or meta_sample.get("batch_size")
    if batch:
        parts.append(f"batch={batch}")
    return " | ".join(str(p) for p in parts if p)


def build_output_path(args: argparse.Namespace, suffix: str, meta_sample: Dict[str, Any] | None = None) -> Path:
    explicit = getattr(args, "output", None)
    if explicit:
        p = Path(explicit)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    save_dir = getattr(args, "save_dir", None)
    if save_dir:
        out_dir = Path(save_dir)
    else:
        out_dir = ROOT / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = args.batch_size or (meta_sample or {}).get("batch_size")
    name_parts = [
        args.scenario,
        args.mode or "MODE",
        args.algo,
        args.api_model or "api",
        f"batch{batch}" if batch else "batch-mixed",
        suffix,
    ]
    return out_dir / ("__".join(str(p) for p in name_parts if p) + ".png")


# =============================================================================
# Display Name Mappings
# =============================================================================

# Canonical ordering for algos and scenarios
ALGO_ORDER = [
    "TournamentExperiment",
    "UtilityExperiment",
    "BaselineRandom",
    "BaselineZscore",
    "FullBatchExperiment",
    "BaselineRerank",
]

SCENARIO_ORDER = ["flights_ithaca_reston", "flights_chi_nyc", "headphones", "exam"]

SCENARIO_DISPLAY_NAMES = {
    "flights_ithaca_reston": "Flights Ithaca->Reston",
    "flights_chi_nyc": "Flights CHI->NYC",
    "headphones": "Headphones",
    "exam": "Exam Scheduling",
}

ALGO_DISPLAY_NAMES = {
    "TournamentExperiment": "LISTEN-T",
    "UtilityExperiment": "LISTEN-U",
    "BaselineRandom": "baseline/random",
    "BaselineZscore": "baseline/zscore-avg",
    "BaselineRerank": "baseline/human-rerank",
    "FullBatchExperiment": "baseline/full-batch",
}

# Color mapping for algorithms
ALGO_COLORS = {
    "TournamentExperiment": "#1f77b4",
    "UtilityExperiment": "#ff7f0e",
    "BaselineRandom": "#2ca02c",
    "BaselineZscore": "#AA3377",
    "BaselineRerank": "#9467bd",
    "FullBatchExperiment": "#d62728",
}


# Marker shape mapping for algorithms
ALGO_MARKERS = {
    "TournamentExperiment": "D",
    "UtilityExperiment": "o",
    "BaselineRandom": "s",
    "BaselineZscore": "^",
    "BaselineRerank": "v",
    "FullBatchExperiment": "X",
}


def get_algo_color(algo_name: str) -> str:
    """Get the color for an algorithm."""
    return ALGO_COLORS.get(algo_name, "gray")


def get_algo_marker(algo_name: str) -> str:
    """Get the marker shape for an algorithm."""
    return ALGO_MARKERS.get(algo_name, "D")


METRIC_DISPLAY_NAMES = {
    "accuracy": "P(chosen = best)",
    "nar": "NAR",
    "gtu": "GTU",
}

# Long-form labels used on y-axes. Titles still use METRIC_DISPLAY_NAMES so
# headers stay short; only the axis label spells the metric out.
METRIC_AXIS_LABELS = {
    "accuracy": "P(chosen = best)",
    "nar": "Normalized Average Rank (mean +/- 2 SE)",
    "gtu": "GTU",
}

FIELD_DISPLAY_NAMES = {
    "batch_size": "Batch Size",
    "api_model": "LLM",
    "algo": "Algorithm",
    "mode": "Mode",
}


def get_algo_display_name(algo_name: str) -> str:
    """Map internal algo name to display name for plots."""
    return ALGO_DISPLAY_NAMES.get(algo_name, algo_name)


def get_scenario_display_name(scenario: str) -> str:
    """Map scenario name to display name for plots."""
    return SCENARIO_DISPLAY_NAMES.get(scenario, scenario)


def get_metric_display_name(metric: str) -> str:
    """Map metric name to short display name for plot titles."""
    return METRIC_DISPLAY_NAMES.get(metric.lower(), metric)


def get_metric_axis_label(metric: str) -> str:
    """Map metric name to long-form y-axis label."""
    return METRIC_AXIS_LABELS.get(metric.lower(), get_metric_display_name(metric))


def get_field_display_name(field: str) -> str:
    """Map field name to display name for axis labels."""
    return FIELD_DISPLAY_NAMES.get(field.lower(), field)


# =============================================================================
# Metric Computation
# =============================================================================

def compute_accuracy(algo: "Experiment") -> float:
    """Compute accuracy: 1 if chosen == best, else 0."""

    # if ranks are same, (ex. exam)
    # algo rank was usually not in human_sol which is why 
    # 

    if algo._winner_idx is None or not algo.human_sol:
        return 0.0
    best_idx = algo.human_sol[0]
    return 1.0 if algo._winner_idx == best_idx else 0.0


def compute_nar(algo: "Experiment") -> "float | None":
    """Compute NAR using Experiment getter."""
    return algo.get_nar()


def compute_gtu(algo: "Experiment") -> "float | None":
    """Compute GTU of the chosen solution."""
    if algo._winner_idx is None:
        return None
    return algo.get_gtu(algo._winner_idx)


METRIC_FUNCTIONS: Dict[str, Callable[["Experiment"], "float | None"]] = {
    "accuracy": compute_accuracy,
    "nar": compute_nar,
    "gtu": compute_gtu,
}


# =============================================================================
# Field Extraction
# =============================================================================

def get_field_value(algo: "Experiment", field: str) -> Any:
    """Extract a field value from an Experiment object."""
    if field == "batch_size":
        return algo.algo_config.get("batch_size") or getattr(algo, "batch_size", None)
    elif field == "api_model":
        return algo.algo_config.get("api_model")
    elif field == "algo":
        return algo.get_algo()
    elif field == "mode":
        return algo.get_mode()
    elif field == "scenario":
        return algo.get_scenario()
    else:
        return algo.algo_config.get(field)


def expand_baseline_variants(algos):
    """Split each baseline into BaselineRandom + BaselineZscore using getters."""
    result = []
    for path, algo in algos:
        if algo.get_algo() != "BaselineAlgorithm":
            result.append((path, algo))
            continue

        algo.algo_label = "BaselineRandom"
        result.append((path, algo))

        zw = algo.get_zscore_winner_idx()
        if zw is not None:
            zs = copy.copy(algo)
            zs.algo_label = "BaselineZscore"
            zs._winner_idx = zw
            result.append((path, zs))

    return result


# =============================================================================
# Rerank Baselines
# =============================================================================

# Maps top20 CSV filename stems to (scenario, mode, scenario_config_file)
_RERANK_FILE_MAP = {
    "Leg 1 Ithaca to Reston VA_numeric_top20": ("flights_ithaca_reston", "Complicated", "flights_ithaca_reston.yml"),
    "Chicago_New York City_combined_numeric_filtered_top20": ("flights_chi_nyc", "Complicated_structured", "flights_chi_nyc.yml"),
    "exam_data_top20": ("exam", "REGISTRAR", "exam.yml"),
    "headphones_data_top20": ("headphones", "MAIN", "headphones.yml"),
}

RERANK_HUMANS = ["h1", "h2", "h3", "h4", "h5"]


def load_rerank_baselines() -> List[Tuple[str, "Experiment"]]:
    """Create synthetic Experiment objects from human rerank top20 CSVs.

    Reads from input/rerank_<human>/*_top20.csv files. Each CSV has a rank
    column and _row_id column. The _row_id of the rank=1 row is the winner.
    """
    import yaml
    import pandas as pd

    configs_dir = ROOT / "configs"
    rerank_dir = ROOT / "input"
    results: List[Tuple[str, Experiment]] = []

    # Cache scenario configs and dataframes
    _scenario_cache: Dict[str, Dict[str, Any]] = {}

    def _get_scenario_data(scenario: str, mode: str, config_file: str):
        cache_key = f"{scenario}_{mode}"
        if cache_key in _scenario_cache:
            return _scenario_cache[cache_key]
        with open(configs_dir / config_file, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        df = pd.read_csv(ROOT / cfg["data_csv"])
        metric_cols = cfg.get("metric_columns", [])
        if metric_cols:
            df = df.dropna(subset=metric_cols)
        gt_human_sol = cfg.get("modes", {}).get(mode, {}).get("human_sol", [])
        gtu_weights = cfg.get("modes", {}).get(mode, {}).get("weights", {})
        _scenario_cache[cache_key] = {
            "scenario": scenario,
            "mode": mode,
            "df": df,
            "metric_columns": metric_cols,
            "gt_human_sol": gt_human_sol,
            "gtu_weights": gtu_weights,
        }
        return _scenario_cache[cache_key]

    # Track (human, scenario) pairs to avoid duplicates from binary/combined variants
    seen: set = set()

    for human in RERANK_HUMANS:
        human_dir = rerank_dir / f"rerank_{human}"
        if not human_dir.exists():
            continue

        for top20_file in sorted(human_dir.glob("*_top20.csv")):
            stem = top20_file.stem
            mapping = _RERANK_FILE_MAP.get(stem)
            if not mapping:
                continue
            scenario, mode, config_file = mapping

            # Skip duplicates (defensive: kept in case future rerank dirs reintroduce variant CSVs that map to the same scenario)
            dedup_key = (human, scenario)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Read the top20 CSV and get the rank-1 row's _row_id
            top20_df = pd.read_csv(top20_file)
            rank1 = top20_df.loc[top20_df["rank"] == 1]
            if rank1.empty:
                continue
            winner_idx = int(rank1.iloc[0]["_row_id"])

            sd = _get_scenario_data(scenario, mode, config_file)

            synthetic = {
                "algo": "baseline",
                "meta": {
                    "scenario": sd["scenario"],
                    "mode": sd["mode"],
                    "algo": "baseline",
                },
                "config": {
                    "scenario": sd["scenario"],
                    "mode": sd["mode"],
                    "metric_columns": sd["metric_columns"],
                    "gtu_weights": sd["gtu_weights"],
                },
                "solutions": sd["df"].to_dict(orient="records"),
                "results": {
                    "winner_idx": winner_idx,
                },
            }
            algo = Experiment.from_json(synthetic)
            algo.algo_label = "BaselineRerank"
            algo._winner_idx = winner_idx
            algo.human_sol = list(sd["gt_human_sol"])

            label = f"rerank_{human}_{scenario}"
            results.append((label, algo))

    return results


# =============================================================================
# Data Loading and Aggregation
# =============================================================================

def load_all_algos(scenario: str, output_dir: "str | None" = None) -> List[Tuple[Path, "Experiment"]]:
    """Load all Experiment objects from a scenario's output directory."""
    base = scenario_outputs_root(scenario, output_dir)
    results = []
    for path in base.glob("**/*.json"):
        try:
            algo = load_algo(path)
            results.append((path, algo))
        except Exception:
            continue
    return results


def aggregate_by_field(
    algos: List[Tuple[Path, "Experiment"]],
    x_field: str,
    y_metric: str,
    group_field: "str | None" = None,
    reps_cap: int | None = 40,
) -> Dict[str, Any]:
    """Aggregate metric values grouped by x_field (and optionally group_field)."""
    metric_fn = METRIC_FUNCTIONS.get(y_metric)
    if not metric_fn:
        raise ValueError(f"Unknown metric: {y_metric}. Available: {list(METRIC_FUNCTIONS.keys())}")

    # Collect: {x_value: {group_value: [y_values]}}
    data: Dict[Any, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(list))

    for _, algo in algos:
        x_val = get_field_value(algo, x_field)
        if x_val is None:
            continue

        y_val = metric_fn(algo)
        if y_val is None:
            continue

        group_val = get_field_value(algo, group_field) if group_field else "_all"
        vals = data[x_val][group_val or "unknown"]
        if reps_cap is None or len(vals) < reps_cap:
            vals.append(y_val)

    if not data:
        raise RuntimeError(f"No data found for x={x_field}, y={y_metric}")

    # Sort x values and groups using canonical orderings when available
    def _sort_key(vals, ordering):
        order_map = {v: i for i, v in enumerate(ordering)}
        return sorted(vals, key=lambda v: order_map.get(v, len(ordering)))

    if x_field == "algo":
        x_values = _sort_key(data.keys(), ALGO_ORDER)
    elif x_field == "scenario":
        x_values = _sort_key(data.keys(), SCENARIO_ORDER)
    else:
        x_values = sorted(data.keys(), key=lambda x: (isinstance(x, str), x))

    all_groups = set()
    for groups in data.values():
        all_groups.update(groups.keys())

    if group_field == "algo":
        group_values = _sort_key(all_groups, ALGO_ORDER)
    elif group_field == "scenario":
        group_values = _sort_key(all_groups, SCENARIO_ORDER)
    else:
        group_values = sorted(all_groups, key=lambda g: (isinstance(g, str), g))

    # Compute mean and 2x stderr using helper
    result = {"x_values": x_values, "x_field": x_field, "y_metric": y_metric, "group_field": group_field, "groups": {}}

    for group_val in group_values:
        means, stderrs, counts = [], [], []
        for x_val in x_values:
            vals = data[x_val].get(group_val, [])
            counts.append(len(vals))
            if not vals:
                means.append(None)
                stderrs.append(None)
            else:
                m, s = compute_mean_and_stderr(vals)
                means.append(m)
                stderrs.append(s)
        result["groups"][group_val] = {"means": means, "stderrs": stderrs, "counts": counts}

    return result
