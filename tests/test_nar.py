#!/usr/bin/env python3
"""Compute Normalized Average Rank (NAR) for flights_chi_nyc.

Given N total items and a human ground-truth ranking of m items (1..m):
  - If selected item is at position p in the list, rank = p
  - If selected item is NOT in the list, rank = (m + 1 + N) / 2
  - NAR = rank / N   (lower is better, in (0, 1])
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "input" / "Chicago_New York City_combined_numeric.csv"
CONFIG_PATH = ROOT / "configs" / "flights_chi_nyc.yml"


def compute_nar(winner_idx, human_sol, N):
    """Compute NAR for a single winner."""
    m = len(human_sol)
    if winner_idx in human_sol:
        rank = human_sol.index(winner_idx) + 1  # 1-indexed position
    else:
        rank = (m + 1 + N) / 2
    return rank / N


def compute_zscore_winner(df, metric_signs):
    """Z-score avg baseline: highest average standardized score."""
    numeric_cols = [col for col, sign in metric_signs.items()
                    if sign != 0 and col in df.columns]
    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    means = numeric_df.mean()
    stds = numeric_df.std().replace(0.0, 1.0)
    z = (numeric_df - means) / stds
    for col in numeric_cols:
        if metric_signs[col] < 0:
            z[col] = -z[col]
    return int(z.mean(axis=1).idxmax())


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(DATA_CSV)
    N = len(df)
    metric_signs = config.get("metric_signs", {})
    modes = config.get("modes", {})

    zscore_winner = compute_zscore_winner(df, metric_signs)
    print(f"Dataset: flights_chi_nyc ({N} items)")
    print(f"Z-score avg winner: idx={zscore_winner}")
    print(f"  -> {df.loc[zscore_winner, 'name']} | ${df.loc[zscore_winner, 'price']} | {df.loc[zscore_winner, 'departure time']}")

    for mode_name, mode_cfg in modes.items():
        human_sol = mode_cfg.get("human_sol")
        if not human_sol:
            continue
        m = len(human_sol)
        nar = compute_nar(zscore_winner, human_sol, N)
        in_list = zscore_winner in human_sol
        rank_str = f"rank {human_sol.index(zscore_winner)+1}/{m}" if in_list else f"unranked (shared rank {(m+1+N)/2:.1f})"
        print(f"\n  Mode: {mode_name}")
        print(f"    human_sol: {human_sol} (m={m})")
        print(f"    zscore winner {rank_str}")
        print(f"    NAR = {nar:.4f}")

        # Also show NAR for the #1 human pick
        top_pick = human_sol[0]
        top_nar = compute_nar(top_pick, human_sol, N)
        print(f"    Best human pick (idx={top_pick}): NAR = {top_nar:.4f}")

    # Assertions
    assert zscore_winner == 472, f"Expected zscore winner idx=472, got {zscore_winner}"
    complicated_human_sol = modes["Complicated"]["human_sol"]
    expected_nar = round(compute_nar(472, complicated_human_sol, N), 4)
    assert expected_nar == 0.5094, f"Expected NAR=0.5094, got {expected_nar}"
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
