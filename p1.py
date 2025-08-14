from batchPref import BatchPrefLearning
import pandas as pd
import numpy as np
import concurrent.futures
import time
import random
from collections import defaultdict, Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Optional, Tuple, Set
from LLM import FreeLLMPreferenceClient
import os 
from Logistic import LinearLogisticModel
from GP import GPUtilityModel
from scipy.stats import norm
from scipy.special import expit

# batch_pref_learning.py
import pandas as pd
import numpy as np
import concurrent.futures
import time
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from sklearn.impute import SimpleImputer

class B2BHeuristicBatchPrefLearning(BatchPrefLearning):
    """
    Deterministic replacement for LLM voting:
    - Winner is the schedule with the lower # of back-to-backs (b2b),
      where b2b = df['other b2b'] + df['evening/morning b2b'].
    - If tied, break by higher model posterior mean (if ready), else lower index.
    - Acquisition / pair selection / model updates remain unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Validate required columns exist
        required_cols = {"other b2b", "evening/morning b2b"}
        missing = required_cols.difference(self.df.columns)
        if missing:
            raise ValueError(
                f"B2BHeuristicBatchPrefLearning requires columns {required_cols}, "
                f"but these are missing from schedules_df: {missing}"
            )

        # Precompute back-to-back counts for all items
        self._b2b = (
            self.df["other b2b"].to_numpy(dtype=float)
            + self.df["evening/morning b2b"].to_numpy(dtype=float)
        )

        # Track the best (lowest) b2b we have seen for optional monitoring
        self.best_b2b_so_far = float("inf")

    def _b2b_of(self, idx: int) -> float:
        return float(self._b2b[idx])

    def _winner_by_b2b(self, idx_a: int, idx_b: int) -> str:
        """
        Return 'A' or 'B' based on lower b2b.
        Tie-breaker:
          1) If utility_model.ready(), pick higher posterior mean util.
          2) Else pick smaller index for determinism.
        """
        b2b_a = self._b2b_of(idx_a)
        b2b_b = self._b2b_of(idx_b)

        if b2b_a < b2b_b:
            return "A"
        if b2b_b < b2b_a:
            return "B"

        # Tie: try model posterior mean (if ready)
        if self.utility_model.ready():
            mu = self.utility_model.posterior_mean_util(self.feat)
            if float(mu[idx_a]) > float(mu[idx_b]):
                return "A"
            if float(mu[idx_b]) > float(mu[idx_a]):
                return "B"

        # Stable fallback
        return "A" if idx_a <= idx_b else "B"

    def _collect_batch_comparisons(self, pairs, prompt_init=None):
        print(f"\n[Heuristic] Collecting comparisons for batch of {len(pairs)} pairs...")
        batch_results, batch_winners = [], []

        for k, (idx_a, idx_b) in enumerate(pairs, start=1):
            winner_side = self._winner_by_b2b(idx_a, idx_b)
            champion = idx_a if winner_side == "A" else idx_b

            # Fabricate deterministic votes
            votes = {"A": self.m_samples if winner_side == "A" else 0,
                     "B": self.m_samples if winner_side == "B" else 0}

            # >>> ADD A SYNTHETIC RESPONSE SO _save_history() HAS SOMETHING TO WRITE
            b2b_a = self._b2b_of(idx_a)
            b2b_b = self._b2b_of(idx_b)
            synthetic_reason = f"Heuristic: lower b2b wins (A={b2b_a:.0f}, B={b2b_b:.0f})"
            responses = [(winner_side, synthetic_reason, None)]
            # <<<

            total_votes = votes["A"] + votes["B"]
            vote_ratio_a = votes["A"] / total_votes if total_votes else 0.5
            vote_ratio_b = votes["B"] / total_votes if total_votes else 0.5

            result = {
                "idx_a": idx_a,
                "idx_b": idx_b,
                "champion_idx": champion,
                "votes": votes,
                "responses": responses,       # <-- now non-empty
                "group_reflection": None,
                "full_prompt": "",
                "total_votes": total_votes,
                "vote_ratio_a": vote_ratio_a,
                "vote_ratio_b": vote_ratio_b,
                "entropy": 0.0,
                "vote_margin": abs(vote_ratio_a - vote_ratio_b),
                "confidence": 1.0,
                "winner": winner_side,
            }
            batch_results.append(result)
            batch_winners.append(champion)

            self._update_cumulative_votes(idx_a, idx_b, votes)
            self.compared_pairs.add((min(idx_a, idx_b), max(idx_a, idx_b)))

        self.previous_winners = list(set(batch_winners))
        return batch_results

    # Optional: make saving robust even if responses were empty
    def _save_history(self):
        rows = []
        for rec in self.history:
            base = {
                "batch_num": rec.get("batch_num"),
                "comparison_num": rec.get("total_comparison_num"),
                "idx_a": rec["idx_a"],
                "idx_b": rec["idx_b"],
                "champion_idx": rec.get("champion_idx"),
                "total_votes": rec.get("total_votes", 1),
                "vote_ratio_a": rec.get("vote_ratio_a", rec["winner"]),
                "vote_ratio_b": rec.get("vote_ratio_b", rec["winner"]),
                "entropy": rec.get("entropy", 0.0),
                "confidence": rec.get("confidence", 1.0),
                "winner": rec["winner"],
            }
            if rec.get("responses"):
                for i, (choice, reason, _reflection) in enumerate(rec["responses"]):
                    row = dict(base)
                    row["voter_id"] = i + 1
                    row["choice"] = choice
                    row["reason"] = reason
                    rows.append(row)
            else:
                # Fallback single row
                row = dict(base)
                row["voter_id"] = 1
                row["choice"] = rec["winner"]
                row["reason"] = "heuristic: lower b2b"
                rows.append(row)

        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)  # ensure directory exists
        history_path = os.path.join(logs_dir, os.path.basename(self.history_file))
        pd.DataFrame(rows).to_csv(history_path, index=False)
        print(f"Saved detailed history to {history_path}")
# Define metrics (same as LLM version)
import os 
import pandas as pd 
from dotenv import load_dotenv

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data.csv")
df = pd.read_csv(data_path)

# Filter out rows with NaN values in utility-relevant columns
utility_columns = ['evening/morning b2b', 'other b2b']
print(f"Original data shape: {df.shape}")
print(f"Rows with NaN in utility columns: {df[utility_columns].isna().any(axis=1).sum()}")

# Remove rows with NaN in utility columns
df = df.dropna(subset=utility_columns)

# Also remove rows with NaN in any metric columns to be safe


print(f"Filtered data shape: {df.shape}")
print(f"Removed {pd.read_csv(data_path).shape[0] - df.shape[0]} rows with NaN values")

metric_columns = [
    "conflicts",
    "quints",
    "quads",
    "four in five slots",
    "triple in 24h (no gaps)",
    "triple in same day (no gaps)",
    "three in four slots",
    "evening/morning b2b",
    "other b2b",
    "two in three slots",
]
df = df.dropna(subset=metric_columns)
# Load environment and create LLM client
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
runner = B2BHeuristicBatchPrefLearning(
    schedules_df=df,
    llm_client=None,                  # not used
    metric_columns=metric_columns,
    model_type="logistic",            # or "gp" if you want
    acq_mode="eubo",
    m_samples=5,
    batch_size=5,
    max_workers=5,
    history_file="b2b_history.csv",

)
final_rankings = runner.run(n_batches=20, save_history=True)
