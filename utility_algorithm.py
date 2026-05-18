from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiment import Experiment
from iteration import AlgorithmIteration, IterationResult


class UtilityIteration(AlgorithmIteration):
    """One refinement step of LISTEN-U."""

    def __init__(
        self,
        experiment,
        iteration_index: int,
        total_iterations: int,
        ctx: Dict[str, Any],
        llm_params: Dict[str, Any],
    ):
        super().__init__(experiment, iteration_index, total_iterations, ctx)
        self.llm_params = llm_params

    def _parse_weights_and_best(self, llm_response: str) -> Tuple[pd.Series, pd.Series]:
        text = llm_response.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        # Strip <END_JSON> tag if present
        text = re.sub(r"\s*<END_JSON>\s*$", "", text)
        if not text:
            raise ValueError("LLM returned an empty response; cannot parse weights")
        response_data = json.loads(text)
        weights = pd.Series(response_data.get("weights", {}))
        prev_iterations = self.ctx.get("utility_history", [])
        if prev_iterations:
            prev_weights = prev_iterations[-1].get("weights")
            if prev_weights is not None:
                for col in self.experiment.numeric_metrics:
                    if col not in weights.index and col in prev_weights.index:
                        weights[col] = prev_weights[col]

        valid_cols = [col for col in self.experiment.numeric_metrics if col in weights.index]
        weights_numeric = weights[valid_cols]

        X = self.experiment.solutions_df[valid_cols].copy()
        col_min = X.min(axis=0)
        col_range = (X.max(axis=0) - col_min).replace(0, np.nan)
        X_norm = X.sub(col_min, axis=1).div(col_range, axis=1).fillna(0.0)
        self.experiment.solutions_df["utility"] = X_norm.dot(weights_numeric)

        best_idx = int(self.experiment.solutions_df["utility"].idxmax())
        best_sol = self.experiment.solutions_df.loc[best_idx]
        return weights_numeric, best_sol

    def execute(self) -> IterationResult:
        pt = self.experiment.prompt_template
        prev_iters = self.ctx.get("utility_history", [])
        if not prev_iters:
            prompt = pt.format(iteration=0)
        else:
            prev = prev_iters[-1]
            prompt = pt.format(
                iteration=self.iteration_index + 1,
                best_solution=prev.get("best_solution"),
                weights=prev.get("weights"),
                formula=prev.get("formula"),
                description=prev.get("description"),
            )

        params = dict(self.llm_params)
        params.setdefault("seed", self.experiment.seed)
        params.setdefault("stop", ["<END_JSON>"])
        response = self.experiment.llm_client.generate_response(prompt, **params)
        parsed = pt.parse_response(response)

        weights_numeric, best_sol = self._parse_weights_and_best(response)
        util_value = float(best_sol.get("utility", 0.0))
        best_idx = int(best_sol.name) if best_sol.name is not None else None

        record = {
            "iteration": self.iteration_index + 1,
            "utility": util_value,
            "winner_idx": best_idx,
            "best_solution": best_sol,
            "weights": weights_numeric,
            "formula": parsed.get("formula") if isinstance(parsed, dict) else None,
            "description": parsed.get("description") if isinstance(parsed, dict) else None,
            "best_utility": util_value,
        }
        print(f"[UtilityExperiment] Iteration {record['iteration']} utility={util_value:.4f} winner_idx={best_idx}")
        self.ctx.setdefault("utility_history", []).append(record)
        self.ctx.setdefault("weights", []).append(weights_numeric)

        algo_specific = {
            "weights": weights_numeric.to_dict(),
            "utility": util_value,
            "formula": record["formula"],
            "description": record["description"],
            "best_solution": best_sol.to_dict(),
        }

        return IterationResult(
            winner_idx=best_idx,
            winner_solution=best_sol,
            stage="utility",
            prompt=prompt,
            response=response,
            nar_delta=1,
            algo_specific=algo_specific,
        )


class UtilityExperiment(Experiment):
    """LISTEN-U implemented with iterations."""

    algo_label = "UtilityExperiment"
    _require_solutions = True
    _require_metrics = True
    _require_llm = True

    def __init__(self, algo_config: Dict[str, Any]) -> None:
        super().__init__(algo_config)

        # Filter numeric_metrics to only columns present in solutions_df
        self.numeric_metrics = [
            col for col in self.numeric_metrics
            if col in self.solutions_df.columns
        ]

        if self.seed is not None:
            random.seed(int(self.seed))
            np.random.seed(int(self.seed))

        self.weights: List[Any] = []

    def make_iteration(self, i: int, total: int, ctx: Dict[str, Any]) -> AlgorithmIteration:
        return UtilityIteration(
            self,
            i,
            total,
            ctx,
            llm_params=self.llm_params,
        )

    def _build_history(self, records: List[IterationResult], start_time: str, end_time: str, ctx: Dict[str, Any]):
        history = super()._build_history(records, start_time, end_time, ctx)
        iterations_list: List[Dict[str, Any]] = []
        weights_list: List[pd.Series] = []

        for idx, (entry, rec) in enumerate(zip(history.get("batch_comparisons", []), records), start=1):
            algo_specific = rec.algo_specific or {}
            entry["stage"] = rec.stage
            if "weights" in algo_specific:
                entry["weights"] = algo_specific["weights"]
                weights_list.append(pd.Series(algo_specific["weights"]))
            if "utility" in algo_specific:
                entry["utility"] = algo_specific["utility"]
            if "best_solution" in algo_specific:
                entry["best_solution"] = algo_specific["best_solution"]
            if "formula" in algo_specific:
                entry["formula"] = algo_specific["formula"]
            if "description" in algo_specific:
                entry["description"] = algo_specific["description"]
            iterations_list.append(
                {
                    "iteration": idx,
                    "utility": algo_specific.get("utility"),
                    "winner_idx": rec.winner_idx,
                    "best_solution": algo_specific.get("best_solution"),
                    "weights": algo_specific.get("weights"),
                    "formula": algo_specific.get("formula"),
                    "description": algo_specific.get("description"),
                    "best_utility": algo_specific.get("utility"),
                }
            )

        history["iterations"] = iterations_list
        self.weights = weights_list
        history.setdefault("metadata", {})["algorithm"] = "utility"
        return history

    def to_json(self) -> Dict[str, Any]:
        payload = super().to_json()
        serialized_weights = []
        for w in self.weights:
            if hasattr(w, "to_dict"):
                serialized_weights.append(w.to_dict())
            else:
                serialized_weights.append(w)
        payload["weights"] = serialized_weights
        return payload

    # ---- Helpers ----------------------------------------------------------

    def get_ith_weights(self, i: int) -> Optional[pd.Series]:
        """Return weights from the i-th iteration (1-based)."""
        if i <= 0:
            return None
        if i - 1 < len(self.weights):
            return self.weights[i - 1]
        if self._history:
            iters = self._history.get("iterations") or []
            if 1 <= i <= len(iters):
                return pd.Series(iters[i - 1].get("weights") or {})
        return None

    def get_normalized_solution_df(self) -> pd.DataFrame:
        """Return a normalized copy of the numeric metrics (no mutation)."""
        if self.solutions_df is None or not self.numeric_metrics:
            return pd.DataFrame()
        X = self.solutions_df[self.numeric_metrics].apply(pd.to_numeric, errors="coerce")
        col_max = X.max(axis=0).replace(0, pd.NA)
        return X.div(col_max, axis=1).fillna(0.0)

    def get_util(self, idx: Any, weights: Any) -> Optional[float]:
        """Compute utility for a solution index given weight mapping/series."""
        try:
            i = int(idx)
        except Exception:
            return None
        if self.solutions_df is None or i < 0 or i >= len(self.solutions_df):
            return None

        w_series = pd.Series(weights)
        w_series = w_series[[c for c in self.numeric_metrics if c in w_series]]
        if w_series.empty:
            return None

        norm_df = self.get_normalized_solution_df()
        if norm_df.empty:
            return None

        util = float(norm_df.loc[i, w_series.index].dot(w_series))
        return util


__all__ = ["UtilityExperiment"]
