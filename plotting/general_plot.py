#!/usr/bin/env python3
"""General plotting script using Experiment object methods with hierarchical CLI.

The CLI uses a hierarchical structure:
- path: Directory containing experiment JSON files
- x_large: Big label for overall graph (generates separate plot per value)
- x_medium: X-axis field
- x_small: Legend/key (color mapping)
- y: Y-axis metric

Examples:
    # NAR vs Scenario, grouped by algorithm
    python general_plot.py --path outputs/flights_ithaca_reston --x_large api_model groq --x_medium scenario --x_small algo --y nar

    # Accuracy vs Batch Size for all algorithms
    python general_plot.py --path outputs/flights_ithaca_reston --x_large algo all --x_medium batch_size --y accuracy

    # GTU vs Algorithm for both api_models (generates one plot per api_model)
    python general_plot.py --path outputs/exam --x_large api_model all --x_medium algo --y gtu
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

plt.rcParams["font.size"] *= 1.4
plt.rcParams["xtick.labelsize"] = 12  # x-tick labels: 1.2× matplotlib default of 10

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotting_helpers import (
    aggregate_by_field,
    expand_baseline_variants,
    load_rerank_baselines,
    get_algo_display_name,
    get_scenario_display_name,
    get_algo_color,
    get_algo_marker,
    get_metric_display_name,
    get_metric_axis_label,
    get_field_display_name,
    get_field_value,
    get_reps_cap,
    load_algo,
    METRIC_FUNCTIONS,
)

# Valid fields for x_large, x_medium, x_small
VALID_FIELDS = {"scenario", "algo", "api_model", "batch_size", "mode"}

# Map CLI shorthand algo names to internal class names
ALGO_CLI_MAP = {
    "tournament": "TournamentExperiment",
    "utility": "UtilityExperiment",
    "baseline": "BaselineAlgorithm",
    "full_batch": "FullBatchExperiment",
}

# Canonical mode per scenario — used with --canonical_mode flag
CANONICAL_MODES: Dict[str, str] = {
    "exam": "REGISTRAR",
    "flights_chi_nyc": "Complicated_structured",
    "flights_ithaca_reston": "Complicated",
    "headphones": "MAIN",
}


def normalize_filter_value(field: str, value: str) -> str:
    """Normalize a CLI filter value to match internal representations."""
    if field == "algo" and value.lower() in ALGO_CLI_MAP:
        return ALGO_CLI_MAP[value.lower()]
    return value


# =============================================================================
# Data Loading
# =============================================================================

def load_algos_from_path(path: Path) -> List[Tuple[Path, Any]]:
    """Load all Experiment objects from a directory."""
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    results = []
    for json_file in path.glob("**/*.json"):
        algo = load_algo(json_file)
        results.append((json_file, algo))
    return results


def get_unique_field_values(algos: List[Tuple[Path, Any]], field: str) -> List[Any]:
    """Get all unique values for a field across loaded experiments."""
    values = set()
    for _, algo in algos:
        val = get_field_value(algo, field)
        if val is not None:
            values.add(val)
    return sorted(values, key=lambda x: (isinstance(x, str), x))


# =============================================================================
# Plotting
# =============================================================================

def _write_n_sidecar(agg_data: Dict[str, Any], plot_path: Path) -> None:
    csv_path = plot_path.with_name(plot_path.stem + "__n.csv")
    x_field = agg_data["x_field"]
    group_field = agg_data["group_field"]
    x_values = agg_data["x_values"]
    groups = agg_data["groups"]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([group_field or "group", x_field, "n", "mean", "stderr"])
        for group_val, group_data in groups.items():
            counts = group_data.get("counts", [])
            means = group_data.get("means", [])
            errs = group_data.get("stderrs", [])
            for i, xv in enumerate(x_values):
                n = counts[i] if i < len(counts) else 0
                m = means[i] if i < len(means) else None
                e = errs[i] if i < len(errs) else None
                writer.writerow([
                    str(group_val),
                    str(xv),
                    str(n),
                    "" if m is None else f"{m:.6f}",
                    "" if e is None else f"{e:.6f}",
                ])
    print(f"Saved sample sizes -> {csv_path}")


def plot_aggregated(
    agg_data: Dict[str, Any],
    title: str,
    output_path: Path,
    show: bool = False,
    filters: Dict[str, Any] | None = None,
) -> None:
    """Plot aggregated data as a dot plot with error bars."""
    x_values = agg_data["x_values"]
    x_field = agg_data["x_field"]
    y_metric = agg_data["y_metric"]
    group_field = agg_data["group_field"]
    groups = agg_data["groups"]

    x_labels = [
        get_algo_display_name(str(x)) if x_field == "algo"
        else get_scenario_display_name(str(x)) if x_field == "scenario"
        else str(x)
        for x in x_values
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    n_groups = len(groups)
    n_x = len(x_values)

    if n_groups == 1 and "_all" in groups:
        # Single-series dot plot. When an algo filter is active, label the
        # series so the legend names the algorithm (e.g. "LISTEN-T") instead
        # of leaving it implicit.
        algo_filter = (filters or {}).get("algo")
        if algo_filter:
            series_label = get_algo_display_name(str(algo_filter))
            series_color = get_algo_color(str(algo_filter))
            series_marker = get_algo_marker(str(algo_filter))
        else:
            series_label = None
            series_color = "steelblue"
            series_marker = "D"

        group_data = groups["_all"]
        valid = [(i, group_data["means"][i], group_data["stderrs"][i])
                 for i in range(len(x_labels)) if group_data["means"][i] is not None]
        if valid:
            indices, means, errs = zip(*valid)
            ax.errorbar(indices, means, yerr=errs, fmt=series_marker, markersize=11,
                        capsize=5, label=series_label, color=series_color,
                        markeredgecolor="black", markeredgewidth=0.5)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        if series_label:
            ax.legend(fontsize=14, loc="upper left", bbox_to_anchor=(0.005, 0.995),
                      frameon=True, framealpha=0.85, borderpad=0.35,
                      labelspacing=0.3, handletextpad=0.45, title=None)
    else:
        # Grouped dot plot with legend
        offset_step = 0.12
        group_spacing = 1.5  # space between x-medium groups
        x_centers = [i * group_spacing for i in range(n_x)]

        for i, (group_val, group_data) in enumerate(groups.items()):
            means = group_data["means"]
            errs = group_data["stderrs"]
            offset = (i - n_groups / 2 + 0.5) * offset_step
            positions = [xc + offset for xc in x_centers]

            # Filter out None values
            valid_positions = []
            valid_means = []
            valid_errs = []
            for pos, m, e in zip(positions, means, errs):
                if m is not None:
                    valid_positions.append(pos)
                    valid_means.append(m)
                    valid_errs.append(e if e is not None else 0)

            # Get display name, color, and marker for algorithm
            display_name = get_algo_display_name(str(group_val)) if group_field == "algo" else str(group_val)
            label = display_name

            # Use specific color and marker for algorithms, otherwise defaults
            color = get_algo_color(str(group_val)) if group_field == "algo" else None
            marker = get_algo_marker(str(group_val)) if group_field == "algo" else "D"

            ax.errorbar(valid_positions, valid_means, yerr=valid_errs, fmt=marker, markersize=11,
                       capsize=4, label=label, markeredgecolor="black", markeredgewidth=0.5,
                       color=color)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(x_labels)
        ax.legend(fontsize=14, loc="upper left", bbox_to_anchor=(0.005, 0.995),
                  frameon=True, framealpha=0.85, borderpad=0.35,
                  labelspacing=0.3, handletextpad=0.45, title=None)

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
    ax.set_xlabel(get_field_display_name(x_field), fontsize=14)
    ax.set_ylabel(get_metric_axis_label(y_metric), fontsize=14)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    _write_n_sidecar(agg_data, output_path)

    if show:
        plt.show()
    plt.close()


def generate_plot_with_filters(
    all_algos: List[Tuple[Path, Any]],
    filters: Dict[str, Any],
    x_medium_field: str,
    x_small_field: str | None,
    y_metric: str,
    output_dir: Path,
    show: bool = False,
    canonical_mode: bool = False,
    per_algo_filters: Dict[str, Dict[str, Any]] | None = None,
    reps_cap: int | None = 40,
) -> None:
    """Generate a single plot with the given filters applied."""
    # Filter algos to those matching ALL filter conditions
    # BaselineRerank and FullBatchExperiment have no api_model in algo_config — let them pass through api_model filters
    _NO_API_MODEL_ALGOS = {"BaselineRerank", "FullBatchExperiment"}
    filtered_algos = []
    for p, a in all_algos:
        match = True
        for field, value in filters.items():
            actual_value = get_field_value(a, field)
            if actual_value is None and field == "api_model" and a.get_algo() in _NO_API_MODEL_ALGOS:
                continue
            if actual_value != value:
                match = False
                break
        if canonical_mode and match:
            scenario = get_field_value(a, "scenario")
            if scenario in CANONICAL_MODES:
                if get_field_value(a, "mode") != CANONICAL_MODES[scenario]:
                    match = False
        if per_algo_filters and match:
            algo_class = a.get_algo()
            if algo_class in per_algo_filters:
                for pf_field, pf_value in per_algo_filters[algo_class].items():
                    actual = get_field_value(a, pf_field)
                    # Coerce string value to match numeric field type
                    coerced = pf_value
                    if isinstance(actual, (int, float)):
                        try:
                            coerced = type(actual)(pf_value)
                        except (ValueError, TypeError):
                            pass
                    if actual != coerced:
                        match = False
                        break
        if match:
            filtered_algos.append((p, a))

    filter_str = ", ".join(f"{f}={v}" for f, v in filters.items())

    if not filtered_algos:
        print(f"No experiments found for {filter_str}")
        return

    print(f"Plotting {len(filtered_algos)} experiments for {filter_str}")

    # Aggregate data
    agg_data = aggregate_by_field(
        filtered_algos, x_medium_field, y_metric, x_small_field,
        reps_cap=reps_cap,
    )

    # Build title from filters
    title_parts = []
    for field, value in filters.items():
        display_val = get_algo_display_name(str(value)) if field == "algo" else str(value)
        title_parts.append(display_val)
    if title_parts:
        title = f"{' | '.join(title_parts)} | {get_metric_display_name(y_metric)} vs {get_field_display_name(x_medium_field)}"
    else:
        title = f"{get_metric_display_name(y_metric)} vs {get_field_display_name(x_medium_field)}"

    # Build filename from filters
    filename_parts = [str(v) for v in filters.values()] if filters else []
    filename_parts.extend([y_metric, x_medium_field])
    if x_small_field:
        filename_parts.append(f"by_{x_small_field}")
    filename = "__".join(filename_parts) + ".png"
    output_path = output_dir / filename

    plot_aggregated(agg_data, title, output_path, show, filters=filters)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate plots from experiment data using hierarchical field structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # NAR vs Scenario, grouped by algorithm
  python general_plot.py --path outputs/flights_ithaca_reston --x_large api_model groq --x_medium scenario --x_small algo --y nar

  # Accuracy vs Batch Size for all algorithms (generates one plot per algorithm)
  python general_plot.py --path outputs/flights_ithaca_reston --x_large algo all --x_medium batch_size --y accuracy

  # GTU vs Algorithm for both api_models (generates two plots)
  python general_plot.py --path outputs/exam --x_large api_model all --x_medium algo --y gtu

Fields: scenario, algo, api_model, batch_size, mode
Metrics: accuracy, nar, gtu
        """
    )
    ap.add_argument(
        "--path", required=True,
        help="Path to directory containing experiment JSON files"
    )
    ap.add_argument(
        "--x_large", nargs='+', metavar="FIELD_VALUE",
        help="Big label: field-value pairs. E.g., 'api_model groq scenario flights_ithaca_reston'. Value can be 'all'. Omit for a single plot."
    )
    ap.add_argument(
        "--x_medium", nargs='+', required=True, metavar="FIELD_VALUE",
        help="X-axis: field and value. E.g., 'scenario all' or 'batch_size 4,8'"
    )
    ap.add_argument(
        "--x_small", nargs='+', metavar="FIELD_VALUE",
        help="Legend/grouping: field name, or field-value pair to filter. E.g., 'algo' or 'algo TournamentAlgorithm'"
    )
    ap.add_argument(
        "--y", required=True, choices=list(METRIC_FUNCTIONS.keys()),
        help="Y-axis metric"
    )
    ap.add_argument(
        "--algo-filter", dest="algo_filter",
        help="Comma-separated list of algorithms to include (e.g., 'tournament,full_batch'). Uses CLI shorthand names."
    )
    ap.add_argument(
        "--extra-path", dest="extra_paths", action="append", default=[],
        help="Additional directory to load experiments from (can be repeated)"
    )
    ap.add_argument(
        "--show", action="store_true",
        help="Display plot interactively"
    )
    ap.add_argument(
        "--canonical_mode", action="store_true",
        help="For each scenario, only include experiments whose mode matches the canonical mode "
             f"(e.g. exam=REGISTRAR, flights_ithaca_reston=Complicated, headphones=MAIN)"
    )
    ap.add_argument(
        "--output-dir", dest="output_dir",
        help="Directory to save plots (default: same as --path)"
    )
    ap.add_argument(
        "--per_algo_filter", nargs=3, action="append", metavar=("ALGO", "FIELD", "VALUE"),
        help="Apply a filter only for a specific algo. E.g., '--per_algo_filter tournament batch_size 8'"
    )
    ap.add_argument(
        "--reps-cap", type=int, default=40,
        help="Maximum runs per plotted cell before aggregation (default: 40; 0 disables)."
    )
    args = ap.parse_args()
    reps_cap = get_reps_cap(args)

    # Parse x_large argument as field-value pairs (optional)
    # E.g., "api_model groq scenario flights_ithaca_reston" -> {api_model: groq, scenario: flights_ithaca_reston}
    x_large_filters: Dict[str, str] = {}
    if args.x_large:
        tokens = args.x_large
        i = 0
        while i < len(tokens):
            field = tokens[i].lower()
            if field not in VALID_FIELDS:
                print(f"Error: Unknown field '{field}'. Valid fields: {', '.join(sorted(VALID_FIELDS))}")
                sys.exit(1)
            if i + 1 >= len(tokens):
                print(f"Error: No value provided for field '{field}'")
                sys.exit(1)
            value = tokens[i + 1]
            x_large_filters[field] = normalize_filter_value(field, value)
            i += 2

    # Parse x_medium argument (must be field-value pair, value can be "all")
    x_medium_tokens = args.x_medium
    if len(x_medium_tokens) < 2:
        print(f"Error: --x_medium requires field and value. E.g., 'scenario all' or 'batch_size 4,8'")
        sys.exit(1)
    x_medium_field = x_medium_tokens[0].lower()
    if x_medium_field not in VALID_FIELDS:
        print(f"Error: Unknown field '{x_medium_field}' for --x_medium. Valid fields: {', '.join(sorted(VALID_FIELDS))}")
        sys.exit(1)
    x_medium_value = x_medium_tokens[1]  # Can be "all" or specific value(s)

    # Parse x_small argument (must be field-value pair, value can be "all")
    x_small_field = None
    x_small_value = None
    if args.x_small:
        tokens = args.x_small
        if len(tokens) < 2:
            print(f"Error: --x_small requires field and value. E.g., 'algo all' or 'algo TournamentAlgorithm'")
            sys.exit(1)
        field = tokens[0].lower()
        if field not in VALID_FIELDS:
            print(f"Error: Unknown field '{field}' for --x_small. Valid fields: {', '.join(sorted(VALID_FIELDS))}")
            sys.exit(1)
        x_small_field = field
        x_small_value = normalize_filter_value(field, tokens[1])  # Can be "all" or specific value

    # Load experiments from path
    data_path = Path(args.path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    all_algos = load_algos_from_path(data_path)

    # Load experiments from extra paths
    for ep in args.extra_paths:
        extra_path = Path(ep)
        if not extra_path.is_absolute():
            extra_path = ROOT / extra_path
        all_algos.extend(load_algos_from_path(extra_path))

    if not all_algos:
        print(f"No experiments found in {data_path}")
        sys.exit(1)

    # Split baselines into random + zscore variants
    all_algos = expand_baseline_variants(all_algos)

    # Add human rerank baselines
    all_algos.extend(load_rerank_baselines())

    # Apply algo filter if specified
    if args.algo_filter:
        allowed = set()
        for name in args.algo_filter.split(","):
            name = name.strip()
            allowed.add(normalize_filter_value("algo", name))
        all_algos = [(p, a) for p, a in all_algos if a.get_algo() in allowed]
        if not all_algos:
            print(f"No experiments match --algo-filter '{args.algo_filter}'")
            sys.exit(1)

    # Filter by x_medium values if not "all"
    x_medium_allowed = None
    if x_medium_value.lower() != "all":
        x_medium_allowed = set(
            normalize_filter_value(x_medium_field, v.strip())
            for v in x_medium_value.split(",")
        )
        all_algos = [
            (p, a) for p, a in all_algos
            if get_field_value(a, x_medium_field) in x_medium_allowed
        ]
        if not all_algos:
            print(f"No experiments match --x_medium {x_medium_field} {x_medium_value}")
            sys.exit(1)

    print(f"Loaded {len(all_algos)} experiments from {data_path}")

    # Parse --per_algo_filter into {AlgoClass: {field: value}}
    per_algo_filters: Dict[str, Dict[str, Any]] = {}
    if args.per_algo_filter:
        for algo_name, field, value in args.per_algo_filter:
            algo_key = normalize_filter_value("algo", algo_name)
            per_algo_filters.setdefault(algo_key, {})[field] = value

    # Collect all fields that need expansion ("all") vs fixed filters
    expand_fields = []
    fixed_filters: Dict[str, str] = {}
    for field, value in x_large_filters.items():
        if value.lower() == "all":
            expand_fields.append(field)
        else:
            fixed_filters[field] = value

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = data_path

    # If x_small has a specific value (not "all"), add it to the fixed filters
    if x_small_value and x_small_value.lower() != "all":
        fixed_filters[x_small_field] = x_small_value

    if expand_fields:
        # Find unique combinations of expand_fields that exist in the data
        from itertools import product as iterproduct

        # Pre-filter algos to those matching fixed filters
        pre_filtered = []
        _NO_API_MODEL_ALGOS = {"BaselineRerank", "FullBatchExperiment"}
        for p, a in all_algos:
            match = True
            for f, v in fixed_filters.items():
                actual = get_field_value(a, f)
                if actual is None and f == "api_model" and a.get_algo() in _NO_API_MODEL_ALGOS:
                    continue
                if actual != v:
                    match = False
                    break
            if match:
                pre_filtered.append((p, a))

        # Get unique values for each expand field (from pre-filtered data)
        expand_value_lists = []
        for ef in expand_fields:
            vals = sorted(set(get_field_value(a, ef) for _, a in pre_filtered if get_field_value(a, ef) is not None),
                          key=lambda x: (isinstance(x, str), x))
            expand_value_lists.append(vals)
            print(f"Found {len(vals)} unique values for {ef}: {vals}")

        # Generate one plot per combination
        for combo in iterproduct(*expand_value_lists):
            filters = dict(fixed_filters)
            for ef, val in zip(expand_fields, combo):
                filters[ef] = val
            generate_plot_with_filters(
                all_algos, filters, x_medium_field, x_small_field, args.y, output_dir, args.show,
                canonical_mode=args.canonical_mode,
                per_algo_filters=per_algo_filters or None,
                reps_cap=reps_cap,
            )
    else:
        # Single plot with all filters applied
        filters = dict(fixed_filters)
        generate_plot_with_filters(
            all_algos, filters, x_medium_field, x_small_field, args.y, output_dir, args.show,
            canonical_mode=args.canonical_mode,
            per_algo_filters=per_algo_filters or None,
            reps_cap=reps_cap,
        )


if __name__ == "__main__":
    main()
