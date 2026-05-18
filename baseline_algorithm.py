from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from experiment import Experiment
from iteration import AlgorithmIteration, IterationResult


class BaselineIteration(AlgorithmIteration):
    """Single iteration of the baseline algorithm (no LLM calls)."""

    def select_random(self, candidates: List[int], seed: Optional[int] = None) -> int:
        if seed is not None:
            random.seed(seed)
        return random.choice(candidates)

    def select_zscore_avg(self) -> int:
        """Pick the solution with the highest average z-score.

        Mirrors the zscore-avg strategy from current-branch: uses metric_signs
        to skip sign=0 columns and flip sign=-1 columns before standardizing,
        then picks the index with the highest average z-score.
        """
        numeric_cols = self.experiment.numeric_metrics
        df = self.experiment.solutions_df
        if df.empty or not numeric_cols:
            return self.select_random(df.index.tolist())

        metric_signs = self.experiment.metric_signs or {}

        # Apply signs: skip sign=0, flip sign=-1
        signed = {}
        for col in numeric_cols:
            if col not in df.columns:
                continue
            sign = int(metric_signs.get(col, -1))
            if sign == 0:
                continue
            val = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            signed[col] = (-1.0 * val) if sign == -1 else val

        if not signed:
            return self.select_random(df.index.tolist())

        signed_df = pd.DataFrame(signed, index=df.index)

        # Z-score standardize
        arr = signed_df.to_numpy(dtype=float)
        means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0)
        stds = np.where(stds == 0.0, 1.0, stds)
        z = (arr - means) / stds
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z_df = pd.DataFrame(z, index=signed_df.index, columns=signed_df.columns)

        avg_z = z_df.mean(axis=1)
        return int(avg_z.idxmax())

    def execute(self) -> IterationResult:
        candidates = self.experiment.solutions_df.index.tolist()

        # Vary seed per iteration so each rep picks a different random solution
        iter_seed = None
        if self.experiment.seed is not None:
            iter_seed = self.experiment.seed + self.iteration_index

        winner_idx = self.select_random(candidates, seed=iter_seed)
        zscore_winner_idx = self.select_zscore_avg()
        winner_solution = self.experiment.solutions_df.iloc[winner_idx].to_dict()

        algo_specific = {
            "batch_indices": candidates,
            "seed": iter_seed,
            "zscore_winner": zscore_winner_idx,
        }

        return IterationResult(
            winner_idx=winner_idx,
            winner_solution=winner_solution,
            stage="single",
            nar_delta=0,
            algo_specific=algo_specific,
        )


class BaselineExperiment(Experiment):
    """Baseline experiment: random selection + z-score ranking (no LLM calls)."""

    algo_label = "BaselineAlgorithm"
    _require_solutions = True
    _require_metrics = False

    def __init__(self, algo_config: Dict[str, Any]) -> None:
        super().__init__(algo_config)

    def make_iteration(self, i: int, total: int, ctx: Dict[str, Any]) -> AlgorithmIteration:
        return BaselineIteration(self, i, total, ctx)

    def get_zscore_winner_idx(self) -> Optional[int]:
        """Return the z-score winner from the first batch comparison."""
        if not self._history:
            return None
        batch_comps = self._history.get("batch_comparisons", [])
        if batch_comps:
            return batch_comps[0].get("zscore_winner")
        return None

    def _build_history(self, records: List[IterationResult], start_time: str, end_time: str, ctx: Dict[str, Any]):
        history: Dict[str, Any] = {
            "metadata": {
                "iterations": self.iterations,
                "n_solutions": len(self.solutions_df),
                "metric_columns": self.metric_columns,
                "start_time": start_time,
                "algorithm": "baseline",
                "end_time": end_time,
            },
            "batch_comparisons": [],
        }

        for rec in records:
            entry = {
                "batch_num": len(history["batch_comparisons"]) + 1,
                "batch_indices": rec.algo_specific.get("batch_indices") if rec.algo_specific else None,
                "winner_idx": rec.winner_idx,
                "seed": (rec.algo_specific or {}).get("seed"),
                "zscore_winner": (rec.algo_specific or {}).get("zscore_winner"),
            }
            history["batch_comparisons"].append(entry)

        return history


# Keep old name as alias for backward compatibility
BaselineAlgorithm = BaselineExperiment

__all__ = ["BaselineExperiment", "BaselineAlgorithm"]
