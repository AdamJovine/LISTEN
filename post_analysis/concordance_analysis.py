#!/usr/bin/env python3
"""
Concordance analysis (paper §4.1 "A Dataset Diagnostic: Concordance").

For each (scenario, mode), estimate the fraction of random linear utility
functions whose argmax solution lands in the human-curated reference set
(`human_sol` in the scenario config).

Procedure:
  1. Sample weight vector w with each component ~ Uniform(-1, 1) over the
     scenario's numeric metrics.
  2. Normalize each numeric attribute to [0, 1] across the full solution set
     and compute u(s) = w · s_norm for every s.
  3. Take s* = argmax u(s) and check whether s* ∈ human_sol.
  4. Repeat n_samples times; report mean ± 2·SE (Bernoulli SE).

A high score => the human reference is supported by a broad slice of utility
weight space (i.e. a linear utility recovers human top picks easily). A low
score => the reference is hard to recover with a linear model.

Ported from LLM-Pref/LISTEN @ origin/current-branch:post_analysis/difficulty_analysis.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_scenario_yaml(scenario: str) -> Path:
    cfg_dir = REPO_ROOT / "configs"
    if scenario.endswith((".yaml", ".yml")):
        p = cfg_dir / scenario
        if p.exists():
            return p
    for ext in (".yml", ".yaml"):
        cand = cfg_dir / f"{scenario}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Scenario '{scenario}' not found under {cfg_dir}")


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def _numeric_min_max(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Tuple[float, float]]:
    min_max: Dict[str, Tuple[float, float]] = {}
    for m in metrics:
        if m not in df.columns:
            continue
        series = pd.to_numeric(df[m], errors="coerce")
        if series.notna().any():
            mn = float(series.min(skipna=True))
            mx = float(series.max(skipna=True))
            min_max[m] = (0.0, 1.0) if mx == mn else (mn, mx)
    return min_max


def normalize_metric(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_utility_scores(
    df: pd.DataFrame,
    weights: Dict[str, float],
    min_max: Dict[str, Tuple[float, float]],
    metric_cols: List[str],
) -> np.ndarray:
    utilities = np.zeros(len(df))
    for metric in metric_cols:
        if metric in weights and metric in df.columns and metric in min_max:
            normalized = normalize_metric(
                pd.to_numeric(df[metric], errors="coerce"),
                min_max[metric][0],
                min_max[metric][1],
            )
            utilities += weights[metric] * normalized.fillna(0.5).values
    return utilities


def _load_human_sol(cfg: dict, mode: str) -> List[int]:
    modes = cfg.get("modes") or {}
    mode_def = modes.get(mode) or {}
    return list(mode_def.get("human_sol") or [])


def measure_concordance(
    scenario: str,
    mode: str,
    n_samples: int = 10000,
    random_seed: Optional[int] = None,
) -> Tuple[float, float]:
    if random_seed is not None:
        np.random.seed(random_seed)

    cfg_path = _find_scenario_yaml(scenario)
    cfg = _load_yaml(cfg_path)

    data_csv = (REPO_ROOT / cfg["data_csv"]).resolve()
    df = pd.read_csv(data_csv)

    metric_cols = list(cfg.get("metric_columns") or [])
    non_numeric = set(cfg.get("non_numeric_metrics") or cfg.get("non_numerical_metrics") or [])
    numeric_metrics = [m for m in metric_cols if m in df.columns and m not in non_numeric]
    min_max = _numeric_min_max(df, numeric_metrics)

    human_sol = set(_load_human_sol(cfg, mode))
    if not human_sol:
        print(f"Warning: no human_sol for {scenario}-{mode}", file=sys.stderr)
        return 0.0, 0.0
    if not numeric_metrics:
        print(f"Warning: no numeric metrics for {scenario}-{mode}", file=sys.stderr)
        return 0.0, 0.0

    matches = 0
    for _ in range(n_samples):
        random_weights = {m: np.random.uniform(-1, 1) for m in numeric_metrics}
        utilities = compute_utility_scores(df, random_weights, min_max, numeric_metrics)
        top_idx = int(np.argmax(utilities))
        if top_idx in human_sol:
            matches += 1

    mean = matches / n_samples
    se = np.sqrt(mean * (1.0 - mean) / n_samples) if n_samples > 1 else 0.0
    return mean, se


def analyze_all_scenarios(
    scenarios: Optional[List[str]] = None,
    n_samples: int = 10000,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    if random_seed is not None:
        np.random.seed(random_seed)

    if scenarios is None:
        cfg_dir = REPO_ROOT / "configs"
        scenarios = sorted(
            {p.stem for p in cfg_dir.glob("*.yml")} | {p.stem for p in cfg_dir.glob("*.yaml")}
        )
        scenarios = [s for s in scenarios if s != "config"]

    results: List[dict] = []
    for scenario in scenarios:
        try:
            cfg = _load_yaml(_find_scenario_yaml(scenario))
        except Exception as e:
            print(f"Error loading {scenario}: {e}", file=sys.stderr)
            continue

        modes = cfg.get("modes") or {}
        for mode_name in modes:
            human_sol = _load_human_sol(cfg, mode_name)
            has_human_sol = bool(human_sol)

            if has_human_sol:
                print(f"Analyzing {scenario}-{mode_name}...")
                mean, se = measure_concordance(scenario, mode_name, n_samples, None)
            else:
                print(f"Skipping {scenario}-{mode_name} (no human_sol)")
                mean = float("nan")
                se = float("nan")

            if np.isnan(mean):
                lower = upper = float("nan")
            else:
                margin = 2.0 * se
                lower = max(0.0, mean - margin)
                upper = min(1.0, mean + margin)

            results.append(
                {
                    "scenario": scenario,
                    "mode": mode_name,
                    "concordance": mean,
                    "concordance_se": se,
                    "concordance_lower": lower,
                    "concordance_upper": upper,
                    "has_human_sol": has_human_sol,
                    "n_human_sol": len(human_sol),
                    "n_samples": n_samples if has_human_sol else 0,
                }
            )

    return pd.DataFrame(results)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Estimate (scenario, mode) concordance via random-weight sampling."
    )
    ap.add_argument("--scenario", type=str, help="Single scenario name")
    ap.add_argument("--mode", type=str, help="Single mode name (requires --scenario)")
    ap.add_argument("--scenarios", nargs="+", help="Subset of scenarios to sweep")
    ap.add_argument("--n-samples", type=int, default=10000)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--output", type=str, help="Optional CSV output path")
    args = ap.parse_args()

    if args.scenario and args.mode:
        mean, se = measure_concordance(args.scenario, args.mode, args.n_samples, args.random_seed)
        margin = 2.0 * se
        lower = max(0.0, mean - margin)
        upper = min(1.0, mean + margin)
        print(
            f"Concordance for {args.scenario}-{args.mode}: "
            f"{mean:.4f} ± {margin:.4f} (95% CI approx. [{lower:.4f}, {upper:.4f}])"
        )
        if args.output:
            pd.DataFrame(
                [
                    {
                        "scenario": args.scenario,
                        "mode": args.mode,
                        "concordance": mean,
                        "concordance_se": se,
                        "concordance_lower": lower,
                        "concordance_upper": upper,
                        "n_samples": args.n_samples,
                    }
                ]
            ).to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        return

    df = analyze_all_scenarios(args.scenarios, args.n_samples, args.random_seed)
    print("\nConcordance Analysis Results:")
    print("=" * 50)
    print(df.to_string(index=False, float_format="%.4f"))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

    valid = df["concordance"].dropna()
    if len(valid):
        print("\nSummary Statistics:")
        print(f"  Mean: {valid.mean():.4f}")
        print(f"  Std:  {valid.std():.4f}")
        print(f"  Min:  {valid.min():.4f}")
        print(f"  Max:  {valid.max():.4f}")


if __name__ == "__main__":
    main()
