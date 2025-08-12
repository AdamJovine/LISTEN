import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Defaults (edit if needed)
# --------------------------
ROOT = "/Users/adamshafikjovine/Documents/LLMPref"
METRICS_CSV_PATH = os.path.join(ROOT, "data.csv")

HIST_A_COL = "idx_a"
HIST_B_COL = "idx_b"
HIST_ITER_COL = "batch_num"
PREF_ONE_MEANS = "B"  # if a 0/1 column is present, 1 means B by default

# --------------------------
# Winner detection (same semantics as before)
# --------------------------
def detect_winner_idx(row):
    cols = row.index

    for c in ["winner_idx", "chosen_idx", "selected_idx"]:
        if c in cols and pd.notna(row[c]):
            return int(row[c])

    if "winner_is_a" in cols and pd.notna(row["winner_is_a"]):
        win_a = bool(row["winner_is_a"])
        return int(row[HIST_A_COL] if win_a else row[HIST_B_COL])

    if "winner" in cols and isinstance(row["winner"], str):
        w = row["winner"].strip().upper()
        if w in {"A", "B"}:
            return int(row[HIST_A_COL] if w == "A" else row[HIST_B_COL])

    for c in ["pref", "choice", "label", "y"]:
        if c in cols and pd.notna(row[c]):
            v = int(row[c])
            if v not in (0, 1):
                raise ValueError(f"Unexpected value {v} in column '{c}', expected 0/1.")
            if v == 1:
                return int(row[HIST_A_COL] if PREF_ONE_MEANS.upper() == "A" else row[HIST_B_COL])
            else:
                return int(row[HIST_B_COL] if PREF_ONE_MEANS.upper() == "A" else row[HIST_A_COL])

    raise ValueError(
        "Could not infer winner. Add one of: "
        "[winner_idx], [chosen_idx], [selected_idx], [winner_is_a], "
        "[winner in {'A','B'}], or a 0/1 column like [pref] and set PREF_ONE_MEANS."
    )

# --------------------------
# Main API
# --------------------------
def make_utility_plot(
    name_patterns,              # e.g. ["50big_batch_pref_historyrandom{i}", "50big_batch_pref_historygp_eubo{i}", ...]
    weights,                    # dict like {"conflicts": -3, ...}
    root=ROOT,
    metrics_csv_path=METRICS_CSV_PATH,
    max_runs=100,
    save_filename="combined_avg_winner_utility_by_iteration.png",
    xlim_max=30,
):
    """
    Build and save a utility plot for multiple methods.

    Parameters
    ----------
    name_patterns : list[str]
        A list of filename patterns with a '{i}' placeholder. Example:
        "50big_batch_pref_historyrandom{i}" ('.csv' optional).
    weights : dict[str, float]
        Mapping from metric name -> weight (penalty negative).
        The keys must exist in metrics.csv (missing keys will be filled as 0).
    root : str
        Base directory containing the history CSVs and metrics CSV.
    metrics_csv_path : str
        Path to the metrics CSV (must include 'schedule_idx' or similar id).
    max_runs : int
        Maximum number of per-method runs to scan (i from 0..max_runs-1).
    save_filename : str
        Output filename for the combined plot (saved under `root`).
    xlim_max : int
        Right edge for x-axis.

    Returns
    -------
    curves : list[tuple[str, pandas.DataFrame]]
        A list of (label, aggregated_df) for each method.
    out_path : str
        Full path to the saved plot.
    """

    # ---- Load metrics once ----
    metrics = pd.read_csv(metrics_csv_path)
    id_col = None
    for cand in ["schedule_idx", "idx", "schedule_id", "id", "Unnamed: 0"]:
        if cand in metrics.columns:
            id_col = cand
            break
    if id_col is None:
        metrics = metrics.reset_index().rename(columns={"index": "schedule_idx"})
        id_col = "schedule_idx"

    metrics = metrics.rename(columns={id_col: "schedule_idx"})
    metrics["schedule_idx"] = pd.to_numeric(metrics["schedule_idx"], errors="coerce")
    metrics = metrics.dropna(subset=["schedule_idx"]).copy()
    metrics["schedule_idx"] = metrics["schedule_idx"].astype(int)

    # Keep only the columns we need; if some weight columns are missing, add them later
    keep_cols = ["schedule_idx"] + list(weights.keys())
    for k in list(weights.keys()):
        if k not in metrics.columns:
            metrics[k] = 0
    metrics = metrics[keep_cols]

    def _label_from_pattern(pat: str) -> str:
        base = os.path.basename(pat)
        base = base.replace("{i}", "")
        if base.endswith(".csv"):
            base = base[:-4]
        return base or pat

    def _pattern_to_path(pat: str, i: int) -> str:
        # Ensure the pattern has {i}; if not, append it
        if "{i}" not in pat:
            pat = pat + "{i}"
        # Ensure .csv
        if not pat.endswith(".csv"):
            pat = pat + ".csv"
        return os.path.join(root, pat.format(i=i))

    def _minmax(col: pd.Series) -> pd.Series | int:
        cmin, cmax = col.min(), col.max()
        return (col - cmin) / (cmax - cmin) if cmax != cmin else 0

    def _load_avg_curve_for_pattern(pattern: str, label: str):
        # Collect existing files
        paths = [_pattern_to_path(pattern, i) for i in range(max_runs)]
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            print(f"[WARN] No files found for {label}. Skipping.")
            return None

        hist_frames_winners = []
        hist_frames_iter0_all = []

        for f in existing:
            dfh = pd.read_csv(f)
            for c in (HIST_ITER_COL, HIST_A_COL, HIST_B_COL):
                if c not in dfh.columns:
                    raise ValueError(f"{os.path.basename(f)} missing required column: {c}")

            dfh = dfh.copy()
            dfh.rename(columns={HIST_ITER_COL: "iteration"}, inplace=True)
            dfh["source_file"] = os.path.basename(f)

            # Winners for iterations > 0 (we'll drop any iteration==0 below)
            dfh["schedule_idx"] = dfh.apply(detect_winner_idx, axis=1)
            winners_only = dfh[["iteration", "source_file", "schedule_idx"]]
            hist_frames_winners.append(winners_only)

            # Iteration-0 baseline from ALL candidates shown in the first batch of this run
            iter0 = dfh["iteration"].min()
            first_batch = dfh[dfh["iteration"] == iter0]
            candidates = pd.unique(
                pd.concat([first_batch[HIST_A_COL], first_batch[HIST_B_COL]], ignore_index=True).dropna()
            )
            iter0_df = pd.DataFrame({
                "iteration": 0,
                "source_file": os.path.basename(f),
                "schedule_idx": candidates.astype(int)
            })
            hist_frames_iter0_all.append(iter0_df)

        winners = pd.concat(hist_frames_winners, ignore_index=True)
        iter0_all = pd.concat(hist_frames_iter0_all, ignore_index=True)

        # Replace any winner rows at iteration 0 with the "all candidates" baseline
        winners = winners[winners["iteration"] != 0]
        combined = pd.concat([iter0_all, winners], ignore_index=True)

        # Merge with metrics; fill missing metric cols with 0
        w = combined.merge(metrics, how="left", on="schedule_idx")
        for m in weights:
            if m not in w.columns:
                w[m] = 0
        w[list(weights.keys())] = w[list(weights.keys())].fillna(0)

        # Normalize per-method across all rows (iter 0 + later winners), then compute weighted utility
        normed = w[list(weights.keys())].apply(_minmax, axis=0)
        w["utility"] = (normed * pd.Series(weights)).sum(axis=1)

        # Aggregate per iteration: mean ± SE
        avg = (
            w.groupby("iteration", as_index=False)
             .agg(
                 avg_winner_utility=("utility", "mean"),
                 se_winner_utility=("utility", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
                 n_samples=("utility", "size"),
                 runs_covered=("source_file", "nunique"),
             )
             .sort_values("iteration")
        )
        return avg

    # Build curves for all patterns
    curves = []
    for pat in name_patterns:
        label = _label_from_pattern(pat)
        avg_df = _load_avg_curve_for_pattern(pat, label)
        if avg_df is not None:
            curves.append((label, avg_df))

    # Plot
    plt.figure(figsize=(9, 5.5))
    marker_cycle = ["o", "s", "D", "^", "v", "p", "X", "*", "h"]
    for idx, (lbl, df) in enumerate(curves):
        marker = marker_cycle[idx % len(marker_cycle)]
        plt.errorbar(
            df["iteration"],
            df["avg_winner_utility"],
            yerr=df["se_winner_utility"],
            fmt=f"{marker}-",
            capsize=4,
            label=lbl,
        )

    plt.xlabel("Iteration (batch_num)")
    plt.ylabel("Average utility ± SE")
    plt.title("Average Utility by Iteration (Iter 0 uses all shown candidates)")
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, xlim_max)
    plt.legend(title="Method")
    plt.tight_layout()

    out_path = os.path.join(root, save_filename)
    plt.savefig(out_path, dpi=200)
    print(f"Saved combined plot: {out_path}")

    return curves, out_path


WEIGHTS = {
    "conflicts": -3,
    "quints": 0,
    "quads": 0,
    "four in five slots": 0,
    "triple in 24h (no gaps)": -2,
    "triple in same day (no gaps)": -2,
    "three in four slots": 0,
    "evening/morning b2b": -1,
    "other b2b": -1,
    "two in three slots": 0,
}

patterns = [
    "50big_batch_pref_historyrandom{i}",
    "50big_batch_pref_historygp_eubo{i}",
    "50big_batch_pref_historylogistic{i}",
]

curves, out_path = make_utility_plot(patterns, WEIGHTS)

# --- Average global rank of winners by iteration ---

# 1) Ensure global utilities & ranks exist for all schedules in `metrics`
if "all_schedules" not in globals() or "global_rank" not in getattr(all_schedules, "columns", []):
    weight_series = pd.Series(WEIGHTS, dtype="float64")
    # Ensure all weighted columns exist
    for c in WEIGHTS:
        if c not in metrics.columns:
            metrics[c] = 0
    all_schedules = metrics[["schedule_idx"] + list(WEIGHTS.keys())].copy()
    all_schedules["utility"] = all_schedules[list(WEIGHTS.keys())].mul(weight_series).sum(axis=1)
    all_schedules["global_rank"] = all_schedules["utility"].rank(method="min", ascending=False).astype(int)

# 2) Attach global rank to winners
winners_ranked = winners.merge(
    all_schedules[["schedule_idx", "global_rank"]],
    on="schedule_idx",
    how="left"
)

# Warn if any winners are missing a rank
missing_rank = winners_ranked["global_rank"].isna().sum()
if missing_rank:
    print(f"[WARN] {missing_rank} winner rows are missing global_rank (schedule not in metrics?). They will be dropped.")
    winners_ranked = winners_ranked.dropna(subset=["global_rank"]).copy()

winners_ranked["global_rank"] = winners_ranked["global_rank"].astype(int)

# 3) Aggregate: average global rank per iteration
avg_rank_by_iter = (
    winners_ranked.groupby("iteration", as_index=False)
                  .agg(avg_global_rank=("global_rank", "mean"),
                       n_winners=("global_rank", "size"))
                  .sort_values("iteration")
)

# Save the table
avg_rank_csv_path = os.path.join(ROOT, "savg_winner_global_rank_per_iteration.csv")
avg_rank_by_iter.to_csv(avg_rank_csv_path, index=False)
print("Saved table:", avg_rank_csv_path)

# 4) Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(
    avg_rank_by_iter["iteration"],
    avg_rank_by_iter["avg_global_rank"],
    marker="o")
plt.xlabel("Iteration")
plt.ylabel("Average global rank of winners (lower is better)")
plt.title("Average Winner Global Rank by Iteration")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0,25)
avg_rank_plot_path = os.path.join(ROOT, "savg_winner_global_rank_by_iteration.png")
plt.savefig(avg_rank_plot_path, dpi=200)
print("Saved plot:", avg_rank_plot_path)

# Optional: also print the top rows for a quick look
print("\nAverage global rank per iteration:")
print(avg_rank_by_iter.head(20).to_string(index=False))
import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Paths and schema
# --------------------------
# /Users/adamshafikjovine/Documents/LLMPref/50big_batch_pref_historygp_eubo4.csv 
ROOT = "/Users/adamshafikjovine/Documents/LLMPref"
HISTORY_FILES = [os.path.join(ROOT, f"50big_batch_pref_historygp_eubo{i}.csv") for i in range(0, 52)]
METRICS_CSV_PATH = os.path.join(ROOT, "data.csv")

HIST_A_COL = "idx_a"
HIST_B_COL = "idx_b"
HIST_ITER_COL = "batch_num"

# Configure how to interpret a 0/1 pref column if present.
# Set to "A" if 1 means A wins; set to "B" if 1 means B wins.
PREF_ONE_MEANS = "B"

# --------------------------
# Weights (penalties)
# --------------------------
WEIGHTS = {
    "conflicts": -3,
    "quints": 0,
    "quads": 0,
    "four in five slots": 0,
    "triple in 24h (no gaps)": -2,
    "triple in same day (no gaps)": -2,
    "three in four slots": 0,
    "evening/morning b2b": -1,
    "other b2b": -1,
    "two in three slots": 0,
}

# --------------------------
# Helper: figure out who won
# --------------------------
def detect_winner_idx(row):
    cols = row.index

    # Case 1: the file already gives the winner's schedule index
    for c in ["winner_idx", "chosen_idx", "selected_idx"]:
        if c in cols and pd.notna(row[c]):
            return int(row[c])

    # Case 2: boolean "winner_is_a"
    if "winner_is_a" in cols and pd.notna(row["winner_is_a"]):
        win_a = bool(row["winner_is_a"])
        return int(row[HIST_A_COL] if win_a else row[HIST_B_COL])

    # Case 3: string label "winner" in {"A","B"}
    if "winner" in cols and isinstance(row["winner"], str):
        w = row["winner"].strip().upper()
        if w in {"A", "B"}:
            return int(row[HIST_A_COL] if w == "A" else row[HIST_B_COL])

    # Case 4: numeric preference 0/1
    for c in ["pref", "choice", "label", "y"]:
        if c in cols and pd.notna(row[c]):
            v = int(row[c])
            if v not in (0, 1):
                raise ValueError(f"Unexpected value {v} in column '{c}', expected 0/1.")
            if v == 1:
                # 1 means A or B depending on config
                return int(row[HIST_A_COL] if PREF_ONE_MEANS.upper() == "A" else row[HIST_B_COL])
            else:  # v == 0
                return int(row[HIST_B_COL] if PREF_ONE_MEANS.upper() == "A" else row[HIST_A_COL])

    # If we get here we couldn't infer the winner — tell the user what to do.
    raise ValueError(
        "Could not infer winner. Add one of: "
        "[winner_idx], [chosen_idx], [selected_idx], [winner_is_a], "
        "[winner in {'A','B'}], or a 0/1 column like [pref] and set PREF_ONE_MEANS."
    )

# --------------------------
# Load histories (exact files)
# --------------------------
existing = [p for p in HISTORY_FILES if os.path.exists(p)]
if not existing:
    raise FileNotFoundError("No specified history files were found.")
missing = sorted(set(HISTORY_FILES) - set(existing))
if missing:
    print("[WARN] Missing files (skipping):")
    for m in missing:
        print("  -", m)

hist_frames = []
for f in existing:
    dfh = pd.read_csv(f)
    for c in (HIST_ITER_COL, HIST_A_COL, HIST_B_COL):
        if c not in dfh.columns:
            raise ValueError(f"{os.path.basename(f)} missing required column: {c}")
    dfh = dfh.copy()
    dfh.rename(columns={HIST_ITER_COL: "iteration"}, inplace=True)
    dfh["source_file"] = os.path.basename(f)

    # produce a single row per comparison holding the WINNER ONLY
    dfh["schedule_idx"] = dfh.apply(detect_winner_idx, axis=1)
    winners_only = dfh[["iteration", "source_file", "schedule_idx"]]
    hist_frames.append(winners_only)

winners = pd.concat(hist_frames, ignore_index=True)

# --------------------------
# Load metrics
# --------------------------
metrics = pd.read_csv(METRICS_CSV_PATH)

# Normalize schedule id column
id_col = None
for cand in ["schedule_idx", "idx", "schedule_id", "id", "Unnamed: 0"]:
    if cand in metrics.columns:
        id_col = cand
        break
if id_col is None:
    metrics = metrics.reset_index().rename(columns={"index": "schedule_idx"})
    id_col = "schedule_idx"

metrics = metrics.rename(columns={id_col: "schedule_idx"})
metrics["schedule_idx"] = pd.to_numeric(metrics["schedule_idx"], errors="coerce")
metrics = metrics.dropna(subset=["schedule_idx"]).copy()
metrics["schedule_idx"] = metrics["schedule_idx"].astype(int)

metrics = metrics[["schedule_idx"] + list(WEIGHTS.keys())]

# --------------------------
# Merge winners with metrics & compute utility
# --------------------------
import numpy as np  # NEW: for standard error calculation

winners = winners.merge(metrics, how="left", on="schedule_idx")
for m in WEIGHTS:
    if m not in winners.columns:
        winners[m] = 0
winners[list(WEIGHTS.keys())] = winners[list(WEIGHTS.keys())].fillna(0)

# --- NEW: normalise each metric to 0-1 before weighting
normed = winners[list(WEIGHTS.keys())].apply(
    lambda col: (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else 0
)

winners["utility"] = (normed * pd.Series(WEIGHTS)).sum(axis=1)

# --------------------------
# Averages (and SE) for winners only
# --------------------------
avg_winners_by_iter = (
    winners.groupby("iteration", as_index=False)
           .agg(
               avg_winner_utility=("utility", "mean"),
               # NEW: Standard error of the mean (SE = SD / sqrt(n))
               se_winner_utility=(
                   "utility",
                   lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 0 else 0
               ),
               n_comparisons=("utility", "size"),
               runs_covered=("source_file", "nunique")
            )
           .sort_values("iteration")
)

overall_avg_winner_utility = winners["utility"].mean()

print("\nAverage **winner** utility per iteration (with SE):")
print(avg_winners_by_iter.head(20))
print(f"\nOverall average utility of winners: {overall_avg_winner_utility:.3f}")

# (Optional) save
avg_winners_by_iter.to_csv(os.path.join(ROOT, "qavg_winner_utility_per_iteration.csv"), index=False)
winners.to_csv(os.path.join(ROOT, "qwinners_with_utility.csv"), index=False)
print("\nSaved:")
print("  -", os.path.join(ROOT, "qavg_winner_utility_per_iteration.csv"))
print("  -", os.path.join(ROOT, "qwinners_with_utility.csv"))

# Ensure sorted by iteration
avg_winners_by_iter = avg_winners_by_iter.sort_values("iteration")

plt.figure(figsize=(8, 5))
# --------------------------
# NEW: Plot with error bars for SE
# --------------------------
plt.errorbar(
    avg_winners_by_iter["iteration"],
    avg_winners_by_iter["avg_winner_utility"],
    yerr=avg_winners_by_iter["se_winner_utility"],
    fmt="o-",  # circle markers connected with lines
    capsize=4,  # add caps to error bars for clarity
)
plt.xlabel("Iteration (batch_num)")
plt.ylabel("Average winner utility ± SE")
plt.title("Average Utility of Winners by Iteration (with Standard Error)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0, 30)
plot_path = os.path.join(ROOT, "savg_winner_utility_by_iteration.png")
plt.savefig(plot_path, dpi=200)
print(f"\nSaved plot: {plot_path}")
import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Paths and schema
# --------------------------
ROOT = "/Users/adamshafikjovine/Documents/LLMPref"
HISTORY_FILES = [os.path.join(ROOT, f"50big_batch_pref_historyrandom{i}.csv") for i in range(0, 52)]
METRICS_CSV_PATH = os.path.join(ROOT, "data.csv")

HIST_A_COL = "idx_a"
HIST_B_COL = "idx_b"
HIST_ITER_COL = "batch_num"

# Configure how to interpret a 0/1 pref column if present.
# Set to "A" if 1 means A wins; set to "B" if 1 means B wins.
PREF_ONE_MEANS = "B"

# --------------------------
# Weights (penalties)
# --------------------------
WEIGHTS = {
    "conflicts": -3,
    "quints": 0,
    "quads": 0,
    "four in five slots": 0,
    "triple in 24h (no gaps)": -2,
    "triple in same day (no gaps)": -2,
    "three in four slots": 0,
    "evening/morning b2b": -1,
    "other b2b": -1,
    "two in three slots": 0,
}

# --------------------------
# Helper: figure out who won
# --------------------------
def detect_winner_idx(row):
    cols = row.index

    # Case 1: the file already gives the winner's schedule index
    for c in ["winner_idx", "chosen_idx", "selected_idx"]:
        if c in cols and pd.notna(row[c]):
            return int(row[c])

    # Case 2: boolean "winner_is_a"
    if "winner_is_a" in cols and pd.notna(row["winner_is_a"]):
        win_a = bool(row["winner_is_a"])
        return int(row[HIST_A_COL] if win_a else row[HIST_B_COL])

    # Case 3: string label "winner" in {"A","B"}
    if "winner" in cols and isinstance(row["winner"], str):
        w = row["winner"].strip().upper()
        if w in {"A", "B"}:
            return int(row[HIST_A_COL] if w == "A" else row[HIST_B_COL])

    # Case 4: numeric preference 0/1
    for c in ["pref", "choice", "label", "y"]:
        if c in cols and pd.notna(row[c]):
            v = int(row[c])
            if v not in (0, 1):
                raise ValueError(f"Unexpected value {v} in column '{c}', expected 0/1.")
            if v == 1:
                # 1 means A or B depending on config
                return int(row[HIST_A_COL] if PREF_ONE_MEANS.upper() == "A" else row[HIST_B_COL])
            else:  # v == 0
                return int(row[HIST_B_COL] if PREF_ONE_MEANS.upper() == "A" else row[HIST_A_COL])

    # If we get here we couldn't infer the winner — tell the user what to do.
    raise ValueError(
        "Could not infer winner. Add one of: "
        "[winner_idx], [chosen_idx], [selected_idx], [winner_is_a], "
        "[winner in {'A','B'}], or a 0/1 column like [pref] and set PREF_ONE_MEANS."
    )

# --------------------------
# Load histories (exact files)
# --------------------------
existing = [p for p in HISTORY_FILES if os.path.exists(p)]
if not existing:
    raise FileNotFoundError("No specified history files were found.")
missing = sorted(set(HISTORY_FILES) - set(existing))
if missing:
    print("[WARN] Missing files (skipping):")
    for m in missing:
        print("  -", m)

hist_frames = []
for f in existing:
    dfh = pd.read_csv(f)
    for c in (HIST_ITER_COL, HIST_A_COL, HIST_B_COL):
        if c not in dfh.columns:
            raise ValueError(f"{os.path.basename(f)} missing required column: {c}")
    dfh = dfh.copy()
    dfh.rename(columns={HIST_ITER_COL: "iteration"}, inplace=True)
    dfh["source_file"] = os.path.basename(f)

    # produce a single row per comparison holding the WINNER ONLY
    dfh["schedule_idx"] = dfh.apply(detect_winner_idx, axis=1)
    winners_only = dfh[["iteration", "source_file", "schedule_idx"]]
    hist_frames.append(winners_only)

winners = pd.concat(hist_frames, ignore_index=True)

# --------------------------
# Load metrics
# --------------------------
metrics = pd.read_csv(METRICS_CSV_PATH)

# Normalize schedule id column
id_col = None
for cand in ["schedule_idx", "idx", "schedule_id", "id", "Unnamed: 0"]:
    if cand in metrics.columns:
        id_col = cand
        break
if id_col is None:
    metrics = metrics.reset_index().rename(columns={"index": "schedule_idx"})
    id_col = "schedule_idx"

metrics = metrics.rename(columns={id_col: "schedule_idx"})
metrics["schedule_idx"] = pd.to_numeric(metrics["schedule_idx"], errors="coerce")
metrics = metrics.dropna(subset=["schedule_idx"]).copy()
metrics["schedule_idx"] = metrics["schedule_idx"].astype(int)

metrics = metrics[["schedule_idx"] + list(WEIGHTS.keys())]

# --------------------------
# Merge winners with metrics & compute utility
# --------------------------
import numpy as np  # NEW: for standard error calculation

winners = winners.merge(metrics, how="left", on="schedule_idx")
for m in WEIGHTS:
    if m not in winners.columns:
        winners[m] = 0
winners[list(WEIGHTS.keys())] = winners[list(WEIGHTS.keys())].fillna(0)

# --- NEW: normalise each metric to 0-1 before weighting
normed = winners[list(WEIGHTS.keys())].apply(
    lambda col: (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else 0
)

winners["utility"] = (normed * pd.Series(WEIGHTS)).sum(axis=1)

# --------------------------
# Averages (and SE) for winners only
# --------------------------
avg_winners_by_iter = (
    winners.groupby("iteration", as_index=False)
           .agg(
               avg_winner_utility=("utility", "mean"),
               # NEW: Standard error of the mean (SE = SD / sqrt(n))
               se_winner_utility=(
                   "utility",
                   lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 0 else 0
               ),
               n_comparisons=("utility", "size"),
               runs_covered=("source_file", "nunique")
            )
           .sort_values("iteration")
)

overall_avg_winner_utility = winners["utility"].mean()

print("\nAverage **winner** utility per iteration (with SE):")
print(avg_winners_by_iter.head(20))
print(f"\nOverall average utility of winners: {overall_avg_winner_utility:.3f}")

# (Optional) save
avg_winners_by_iter.to_csv(os.path.join(ROOT, "qavg_winner_utility_per_iteration.csv"), index=False)
winners.to_csv(os.path.join(ROOT, "qwinners_with_utility.csv"), index=False)
print("\nSaved:")
print("  -", os.path.join(ROOT, "qavg_winner_utility_per_iteration.csv"))
print("  -", os.path.join(ROOT, "qwinners_with_utility.csv"))

# Ensure sorted by iteration
avg_winners_by_iter = avg_winners_by_iter.sort_values("iteration")

plt.figure(figsize=(8, 5))
# --------------------------
# NEW: Plot with error bars for SE
# --------------------------
plt.errorbar(
    avg_winners_by_iter["iteration"],
    avg_winners_by_iter["avg_winner_utility"],
    yerr=avg_winners_by_iter["se_winner_utility"],
    fmt="o-",  # circle markers connected with lines
    capsize=4,  # add caps to error bars for clarity
)
plt.xlabel("Iteration (batch_num)")
plt.ylabel("Average winner utility ± SE")
plt.title("Average Utility of Winners by Iteration (with Standard Error)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0, 30)
plot_path = os.path.join(ROOT, "savg_winner_utility_by_iteration.png")
plt.savefig(plot_path, dpi=200)
print(f"\nSaved plot: {plot_path}")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Paths and schema (edit ROOT if needed)
# --------------------------
ROOT = "/Users/adamshafikjovine/Documents/LLMPref"
METRICS_CSV_PATH = os.path.join(ROOT, "data.csv")

HIST_A_COL = "idx_a"
HIST_B_COL = "idx_b"
HIST_ITER_COL = "batch_num"
PREF_ONE_MEANS = "B"  # if a 0/1 column is present, 1 means B by default

# --------------------------
# Weights (penalties)
# --------------------------
WEIGHTS = {
    "conflicts": -3,
    "quints": 0,
    "quads": 0,
    "four in five slots": 0,
    "triple in 24h (no gaps)": -2,
    "triple in same day (no gaps)": -2,
    "three in four slots": 0,
    "evening/morning b2b": -1,
    "other b2b": -1,
    "two in three slots": 0,
}

# --------------------------
# Helper: figure out who won (same logic as your scripts)
# --------------------------
def detect_winner_idx(row):
    cols = row.index

    for c in ["winner_idx", "chosen_idx", "selected_idx"]:
        if c in cols and pd.notna(row[c]):
            return int(row[c])

    if "winner_is_a" in cols and pd.notna(row["winner_is_a"]):
        win_a = bool(row["winner_is_a"])
        return int(row[HIST_A_COL] if win_a else row[HIST_B_COL])

    if "winner" in cols and isinstance(row["winner"], str):
        w = row["winner"].strip().upper()
        if w in {"A", "B"}:
            return int(row[HIST_A_COL] if w == "A" else row[HIST_B_COL])

    for c in ["pref", "choice", "label", "y"]:
        if c in cols and pd.notna(row[c]):
            v = int(row[c])
            if v not in (0, 1):
                raise ValueError(f"Unexpected value {v} in column '{c}', expected 0/1.")
            if v == 1:
                return int(row[HIST_A_COL] if PREF_ONE_MEANS.upper() == "A" else row[HIST_B_COL])
            else:
                return int(row[HIST_B_COL] if PREF_ONE_MEANS.upper() == "A" else row[HIST_A_COL])

    raise ValueError(
        "Could not infer winner. Add one of: "
        "[winner_idx], [chosen_idx], [selected_idx], [winner_is_a], "
        "[winner in {'A','B'}], or a 0/1 column like [pref] and set PREF_ONE_MEANS."
    )

# --------------------------
# Load metrics once
# --------------------------
metrics = pd.read_csv(METRICS_CSV_PATH)
id_col = None
for cand in ["schedule_idx", "idx", "schedule_id", "id", "Unnamed: 0"]:
    if cand in metrics.columns:
        id_col = cand
        break
if id_col is None:
    metrics = metrics.reset_index().rename(columns={"index": "schedule_idx"})
    id_col = "schedule_idx"

metrics = metrics.rename(columns={id_col: "schedule_idx"})
metrics["schedule_idx"] = pd.to_numeric(metrics["schedule_idx"], errors="coerce")
metrics = metrics.dropna(subset=["schedule_idx"]).copy()
metrics["schedule_idx"] = metrics["schedule_idx"].astype(int)
metrics = metrics[["schedule_idx"] + list(WEIGHTS.keys())]

# --------------------------
# Pipeline for one method
# --------------------------
def load_avg_curve(file_pattern_func, label):
    # Gather existing files
    history_files = [file_pattern_func(i) for i in range(0, 52)]
    existing = [p for p in history_files if os.path.exists(p)]
    if not existing:
        print(f"[WARN] No files found for {label}. Skipping.")
        return None, None

    hist_frames = []
    for f in existing:
        dfh = pd.read_csv(f)
        for c in (HIST_ITER_COL, HIST_A_COL, HIST_B_COL):
            if c not in dfh.columns:
                raise ValueError(f"{os.path.basename(f)} missing required column: {c}")
        dfh = dfh.copy()
        dfh.rename(columns={HIST_ITER_COL: "iteration"}, inplace=True)
        dfh["source_file"] = os.path.basename(f)
        dfh["schedule_idx"] = dfh.apply(detect_winner_idx, axis=1)
        winners_only = dfh[["iteration", "source_file", "schedule_idx"]]
        hist_frames.append(winners_only)

    winners = pd.concat(hist_frames, ignore_index=True)

    # Merge with metrics
    w = winners.merge(metrics, how="left", on="schedule_idx")
    for m in WEIGHTS:
        if m not in w.columns:
            w[m] = 0
    w[list(WEIGHTS.keys())] = w[list(WEIGHTS.keys())].fillna(0)

    # Normalize metric columns to [0,1] then weight-sum into utility
    normed = w[list(WEIGHTS.keys())].apply(
        lambda col: (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else 0
    )
    w["utility"] = (normed * pd.Series(WEIGHTS)).sum(axis=1)

    # Aggregate per iteration (mean + SE)
    avg = (
        w.groupby("iteration", as_index=False)
         .agg(
             avg_winner_utility=("utility", "mean"),
             se_winner_utility=("utility", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 0 else 0),
             n_comparisons=("utility", "size"),
             runs_covered=("source_file", "nunique"),
         )
         .sort_values("iteration")
    )
    return label, avg

# --------------------------
# Define three methods
# --------------------------
cfgs = [
    ("random",   lambda i: os.path.join(ROOT, f"50big_batch_pref_historyrandom{i}.csv")),
    ("gp_eubo",  lambda i: os.path.join(ROOT, f"50big_batch_pref_historygp_eubo{i}.csv")),
    ("logistic", lambda i: os.path.join(ROOT, f"50big_batch_pref_historylogistic{i}.csv")),
]

curves = []
for label, fn in cfgs:
    lbl, df = load_avg_curve(fn, label)
    if df is not None:
        curves.append((lbl, df))

# --------------------------
# Plot: all three on one figure
# --------------------------
plt.figure(figsize=(9, 5.5))
marker_map = {"random": "o", "gp_eubo": "s", "logistic": "D"}
for lbl, df in curves:
    plt.errorbar(
        df["iteration"],
        df["avg_winner_utility"],
        yerr=df["se_winner_utility"],
        fmt=f"{marker_map.get(lbl, 'o')}-",
        capsize=4,
        label=lbl,
    )

plt.xlabel("Iteration (batch_num)")
plt.ylabel("Average winner utility ± SE")
plt.title("Average Utility of Winners by Iteration")
plt.grid(True, alpha=0.3)
plt.xlim(0, 30)
plt.legend(title="Method")
plt.tight_layout()

out_path = os.path.join(ROOT, "combined_avg_winner_utility_by_iteration.png")
plt.savefig(out_path, dpi=200)
print(f"Saved combined plot: {out_path}")


WEIGHTS = {
    "conflicts": 0,
    "quints": 0,
    "quads": 0,
    "four in five slots": 0,
    "triple in 24h (no gaps)": 0,
    "triple in same day (no gaps)": 0,
    "three in four slots": 0,
    "evening/morning b2b": -1,
    "other b2b": -1,
    "two in three slots": 0,
}


# --------------------------
# Load metrics once
# --------------------------
metrics = pd.read_csv(METRICS_CSV_PATH)
id_col = None
for cand in ["schedule_idx", "idx", "schedule_id", "id", "Unnamed: 0"]:
    if cand in metrics.columns:
        id_col = cand
        break
if id_col is None:
    metrics = metrics.reset_index().rename(columns={"index": "schedule_idx"})
    id_col = "schedule_idx"

metrics = metrics.rename(columns={id_col: "schedule_idx"})
metrics["schedule_idx"] = pd.to_numeric(metrics["schedule_idx"], errors="coerce")
metrics = metrics.dropna(subset=["schedule_idx"]).copy()
metrics["schedule_idx"] = metrics["schedule_idx"].astype(int)
metrics = metrics[["schedule_idx"] + list(WEIGHTS.keys())]

# --------------------------
# Pipeline for one method
# --------------------------
def load_avg_curve(file_pattern_func, label):
    # Gather existing files
    history_files = [file_pattern_func(i) for i in range(0, 52)]
    existing = [p for p in history_files if os.path.exists(p)]
    if not existing:
        print(f"[WARN] No files found for {label}. Skipping.")
        return None, None

    hist_frames = []
    for f in existing:
        dfh = pd.read_csv(f)
        for c in (HIST_ITER_COL, HIST_A_COL, HIST_B_COL):
            if c not in dfh.columns:
                raise ValueError(f"{os.path.basename(f)} missing required column: {c}")
        dfh = dfh.copy()
        dfh.rename(columns={HIST_ITER_COL: "iteration"}, inplace=True)
        dfh["source_file"] = os.path.basename(f)
        dfh["schedule_idx"] = dfh.apply(detect_winner_idx, axis=1)
        winners_only = dfh[["iteration", "source_file", "schedule_idx"]]
        hist_frames.append(winners_only)

    winners = pd.concat(hist_frames, ignore_index=True)

    # Merge with metrics
    w = winners.merge(metrics, how="left", on="schedule_idx")
    for m in WEIGHTS:
        if m not in w.columns:
            w[m] = 0
    w[list(WEIGHTS.keys())] = w[list(WEIGHTS.keys())].fillna(0)

    # Normalize metric columns to [0,1] then weight-sum into utility
    normed = w[list(WEIGHTS.keys())].apply(
        lambda col: (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else 0
    )
    w["utility"] = (normed * pd.Series(WEIGHTS)).sum(axis=1)

    # Aggregate per iteration (mean + SE)
    avg = (
        w.groupby("iteration", as_index=False)
         .agg(
             avg_winner_utility=("utility", "mean"),
             se_winner_utility=("utility", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 0 else 0),
             n_comparisons=("utility", "size"),
             runs_covered=("source_file", "nunique"),
         )
         .sort_values("iteration")
    )
    return label, avg

# --------------------------
# Define three methods
# --------------------------
cfgs = [
    ("original",   lambda i: os.path.join(ROOT, f"50big_batch_pref_historyrandom{i}.csv")),
    ("minB2bwithRef",  lambda i: os.path.join(ROOT, f"50big_batch_pref_historygp_eubo{i}prompt0.csv")),
    ("minB2bnoRef", lambda i: os.path.join(ROOT, f"50big_batch_pref_historylogistic{i}.csv")),
]

curves = []
for label, fn in cfgs:
    lbl, df = load_avg_curve(fn, label)
    if df is not None:
        curves.append((lbl, df))

# --------------------------
# Plot: all three on one figure
# --------------------------
plt.figure(figsize=(9, 5.5))
marker_map = {"random": "o", "gp_eubo": "s", "logistic": "D"}
for lbl, df in curves:
    plt.errorbar(
        df["iteration"],
        df["avg_winner_utility"],
        yerr=df["se_winner_utility"],
        fmt=f"{marker_map.get(lbl, 'o')}-",
        capsize=4,
        label=lbl,
    )

plt.xlabel("Iteration (batch_num)")
plt.ylabel("Average winner utility ± SE")
plt.title("Average Utility of Winners by Iteration")
plt.grid(True, alpha=0.3)
plt.xlim(0, 30)
plt.legend(title="Method")
plt.tight_layout()

out_path = os.path.join(ROOT, "combined_avg_winner_utility_by_iteration.png")
plt.savefig(out_path, dpi=200)
print(f"Saved combined plot: {out_path}")
def load_avg_curve(file_pattern_func, label):
  # Gather existing files
  history_files = [file_pattern_func(i) for i in range(0, 52)]
  existing = [p for p in history_files if os.path.exists(p)]
  if not existing:
    print(f"[WARN] No files found for {label}. Skipping.")
    return None, None

  hist_frames = []
  for f in existing:
    dfh = pd.read_csv(f)
    for c in (HIST_ITER_COL, HIST_A_COL, HIST_B_COL):
      if c not in dfh.columns:
        raise ValueError(f"{os.path.basename(f)} missing required column: {c}")
    dfh = dfh.copy()
    dfh.rename(columns={HIST_ITER_COL: "iteration"}, inplace=True)
    dfh["source_file"] = os.path.basename(f)

    # If iteration is 0, duplicate the row for both candidates
    df0 = dfh[dfh["iteration"] == 0].copy()
    if not df0.empty:
      df0_a = df0.copy()
      df0_a["schedule_idx"] = df0_a[HIST_A_COL]
      df0_b = df0.copy()
      df0_b["schedule_idx"] = df0_b[HIST_B_COL]
      df0_long = pd.concat([df0_a, df0_b], ignore_index=True)
    else:
      df0_long = pd.DataFrame()

    # For all other iterations, use the winner as determined by detect_winner_idx
    df_non0 = dfh[dfh["iteration"] != 0].copy()
    if not df_non0.empty:
      df_non0["schedule_idx"] = df_non0.apply(detect_winner_idx, axis=1)

    winners_only = pd.concat([df0_long, df_non0], ignore_index=True)
    winners_only = winners_only[["iteration", "source_file", "schedule_idx"]]
    hist_frames.append(winners_only)

  winners = pd.concat(hist_frames, ignore_index=True)

  # Merge with metrics
  w = winners.merge(metrics, how="left", on="schedule_idx")
  for m in WEIGHTS:
    if m not in w.columns:
      w[m] = 0
  w[list(WEIGHTS.keys())] = w[list(WEIGHTS.keys())].fillna(0)

  # Normalize metric columns to [0,1] then weight-sum into utility
  normed = w[list(WEIGHTS.keys())].apply(
    lambda col: (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else 0
  )
  w["utility"] = (normed * pd.Series(WEIGHTS)).sum(axis=1)

  # Aggregate per iteration (mean + SE)
  avg = (
    w.groupby("iteration", as_index=False)
     .agg(
       avg_winner_utility=("utility", "mean"),
       se_winner_utility=("utility", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 0 else 0),
       n_comparisons=("utility", "size"),
       runs_covered=("source_file", "nunique"),
     )
     .sort_values("iteration")
  )
  return label, avg

import os

if __name__ == "__main__":
  # Ensure ROOT is defined. Adjust as needed.
  ROOT = "/Users/adamshafikjovine/Documents/LLMPref"
  
  # --------------------------
  # Define three methods
  # --------------------------
  cfgs = [
    ("random",   lambda i: os.path.join(ROOT, f"50big_batch_pref_historyrandom{i}.csv")),
    ("gp_eubo",  lambda i: os.path.join(ROOT, f"50big_batch_pref_historygp_eubo{i}.csv")),
    ("logistic", lambda i: os.path.join(ROOT, f"50big_batch_pref_historylogistic{i}.csv")),
  ]
  
  curves = []
  for label, file_pattern_func in cfgs:
    lbl, avg = load_avg_curve(file_pattern_func, label)
    if avg is not None:
      curves.append((lbl, avg))
    else:
      print(f"[INFO] No data for {label}")
  
  # For demonstration, print the first few rows of the computed averages
  for label, avg in curves:
    print(f"Results for {label}:")
    print(avg.head())

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Paths and schema (edit ROOT if needed)
# --------------------------
ROOT = "/Users/adamshafikjovine/Documents/LLMPref"
METRICS_CSV_PATH = os.path.join(ROOT, "data.csv")

HIST_A_COL = "idx_a"
HIST_B_COL = "idx_b"
HIST_ITER_COL = "batch_num"
PREF_ONE_MEANS = "B"  # if a 0/1 column is present, 1 means B by default

# --------------------------
# Weights (penalties)
# --------------------------
WEIGHTS = {
    "conflicts": -3,
    "quints": 0,
    "quads": 0,
    "four in five slots": 0,
    "triple in 24h (no gaps)": -2,
    "triple in same day (no gaps)": -2,
    "three in four slots": 0,
    "evening/morning b2b": -1,
    "other b2b": -1,
    "two in three slots": 0,
}

# --------------------------
# Helper: figure out who won (same logic as your scripts)
# --------------------------
def detect_winner_idx(row):
    cols = row.index

    for c in ["winner_idx", "chosen_idx", "selected_idx"]:
        if c in cols and pd.notna(row[c]):
            return int(row[c])

    if "winner_is_a" in cols and pd.notna(row["winner_is_a"]):
        win_a = bool(row["winner_is_a"])
        return int(row[HIST_A_COL] if win_a else row[HIST_B_COL])

    if "winner" in cols and isinstance(row["winner"], str):
        w = row["winner"].strip().upper()
        if w in {"A", "B"}:
            return int(row[HIST_A_COL] if w == "A" else row[HIST_B_COL])

    for c in ["pref", "choice", "label", "y"]:
        if c in cols and pd.notna(row[c]):
            v = int(row[c])
            if v not in (0, 1):
                raise ValueError(f"Unexpected value {v} in column '{c}', expected 0/1.")
            if v == 1:
                return int(row[HIST_A_COL] if PREF_ONE_MEANS.upper() == "A" else row[HIST_B_COL])
            else:
                return int(row[HIST_B_COL] if PREF_ONE_MEANS.upper() == "A" else row[HIST_A_COL])

    raise ValueError(
        "Could not infer winner. Add one of: "
        "[winner_idx], [chosen_idx], [selected_idx], [winner_is_a], "
        "[winner in {'A','B'}], or a 0/1 column like [pref] and set PREF_ONE_MEANS."
    )

# --------------------------
# Load metrics once
# --------------------------
metrics = pd.read_csv(METRICS_CSV_PATH)
id_col = None
for cand in ["schedule_idx", "idx", "schedule_id", "id", "Unnamed: 0"]:
    if cand in metrics.columns:
        id_col = cand
        break
if id_col is None:
    metrics = metrics.reset_index().rename(columns={"index": "schedule_idx"})
    id_col = "schedule_idx"

metrics = metrics.rename(columns={id_col: "schedule_idx"})
metrics["schedule_idx"] = pd.to_numeric(metrics["schedule_idx"], errors="coerce")
metrics = metrics.dropna(subset=["schedule_idx"]).copy()
metrics["schedule_idx"] = metrics["schedule_idx"].astype(int)
metrics = metrics[["schedule_idx"] + list(WEIGHTS.keys())]

# --------------------------
# Pipeline for one method
# --------------------------
def load_avg_curve(file_pattern_func, label):
    # Gather existing files
    history_files = [file_pattern_func(i) for i in range(0, 52)]
    existing = [p for p in history_files if os.path.exists(p)]
    if not existing:
        print(f"[WARN] No files found for {label}. Skipping.")
        return None, None

    hist_frames = []
    for f in existing:
        dfh = pd.read_csv(f)
        for c in (HIST_ITER_COL, HIST_A_COL, HIST_B_COL):
            if c not in dfh.columns:
                raise ValueError(f"{os.path.basename(f)} missing required column: {c}")
        dfh = dfh.copy()
        dfh.rename(columns={HIST_ITER_COL: "iteration"}, inplace=True)
        dfh["source_file"] = os.path.basename(f)
        dfh["schedule_idx"] = dfh.apply(detect_winner_idx, axis=1)
        winners_only = dfh[["iteration", "source_file", "schedule_idx"]]
        hist_frames.append(winners_only)

    winners = pd.concat(hist_frames, ignore_index=True)

    # Merge with metrics
    w = winners.merge(metrics, how="left", on="schedule_idx")
    for m in WEIGHTS:
        if m not in w.columns:
            w[m] = 0
    w[list(WEIGHTS.keys())] = w[list(WEIGHTS.keys())].fillna(0)

    # Normalize metric columns to [0,1] then weight-sum into utility
    normed = w[list(WEIGHTS.keys())].apply(
        lambda col: (col - col.min()) / (col.max() - col.min()) if col.max() != col.min() else 0
    )
    w["utility"] = (normed * pd.Series(WEIGHTS)).sum(axis=1)

    # Aggregate per iteration (mean + SE)
    avg = (
        w.groupby("iteration", as_index=False)
         .agg(
             avg_winner_utility=("utility", "mean"),
             se_winner_utility=("utility", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 0 else 0),
             n_comparisons=("utility", "size"),
             runs_covered=("source_file", "nunique"),
         )
         .sort_values("iteration")
    )
    return label, avg

# --------------------------
# Define three methods
# --------------------------
cfgs = [
    ("random",   lambda i: os.path.join(ROOT, f"newbig_batch_pref_historyrandom{i}prompt0historyFalse.csv")),
    ("gp_eubo",  lambda i: os.path.join(ROOT, f"newbig_batch_pref_historygp_eubo{i}prompt0historyFalse.csv")),
    ("logistic", lambda i: os.path.join(ROOT, f"newbig_batch_pref_historylogistic{i}prompt0historyFalse.csv")),
]

curves = []
for label, fn in cfgs:
    lbl, df = load_avg_curve(fn, label)
    if df is not None:
        curves.append((lbl, df))

# --------------------------
# Plot: all three on one figure
# --------------------------
plt.figure(figsize=(9, 5.5))
marker_map = {"random": "o", "gp_eubo": "s", "logistic": "D"}
for lbl, df in curves:
    plt.errorbar(
        df["iteration"],
        df["avg_winner_utility"],
        yerr=df["se_winner_utility"],
        fmt=f"{marker_map.get(lbl, 'o')}-",
        capsize=4,
        label=lbl,
    )

plt.xlabel("Iteration (batch_num)")
plt.ylabel("Average winner utility ± SE")
plt.title("Average Utility of Winners by Iteration")
plt.grid(True, alpha=0.3)
plt.xlim(0, 30)
plt.legend(title="Method")
plt.tight_layout()

out_path = os.path.join(ROOT, "combined_avg_winner_utility_by_iteration.png")
plt.savefig(out_path, dpi=200)
print(f"Saved combined plot: {out_path}")
