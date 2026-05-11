#!/usr/bin/env python3
"""Compute the z-score average winner for flights_chi_nyc.

Method: For each numerical attribute, standardize values to zero mean and unit
variance (z-score). Negate scores for attributes to be minimized. Select the
item with the highest average standardized score across all numerical attributes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "input" / "Chicago_New York City_combined_numeric.csv"

# From configs/flights_chi_nyc.yml — metric_sign of -1 means minimize, 0 means ignore
METRIC_SIGNS = {
    "name": 0,
    "origin": 0,
    "destination": 0,
    "departure time": 0,
    "arrival time": 0,
    "duration": 0,
    "stops": -1,
    "price": -1,
    "dis_from_origin": -1,
    "dis_from_dest": -1,
    "departure_seconds": -1,
    "arrival_seconds": 0,
    "duration_min": -1,
}


def main():
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} items from {DATA_CSV.name}")

    # Keep only numeric columns with non-zero metric_sign
    numeric_cols = [col for col, sign in METRIC_SIGNS.items()
                    if sign != 0 and col in df.columns]
    print(f"Numeric columns used: {numeric_cols}")

    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Z-score standardize
    means = numeric_df.mean()
    stds = numeric_df.std()
    stds = stds.replace(0.0, 1.0)
    z = (numeric_df - means) / stds

    # Negate columns that should be minimized
    for col in numeric_cols:
        if METRIC_SIGNS[col] < 0:
            z[col] = -z[col]

    # Average z-score per item
    avg_z = z.mean(axis=1)
    winner_idx = int(avg_z.idxmax())

    print(f"\nWinner: row {winner_idx} (avg z-score = {avg_z.loc[winner_idx]:.4f})")
    print(f"\nWinner details:")
    for col in df.columns:
        print(f"  {col}: {df.loc[winner_idx, col]}")

    print(f"\nTop 10 by avg z-score:")
    for idx, val in avg_z.nlargest(10).items():
        row = df.loc[idx]
        print(f"  idx={idx}: avg_z={val:.4f}  price={row['price']}  stops={row['stops']}  duration_min={row['duration_min']}  {row['name']}")


if __name__ == "__main__":
    main()
