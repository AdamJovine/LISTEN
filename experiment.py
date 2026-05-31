from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from iteration import AlgorithmIteration, IterationResult


class _NullLLMClient:
    """Placeholder client used when restoring from JSON without a real LLM client."""

    def generate_response(self, prompt: str, **kwargs):
        raise RuntimeError(
            "This algorithm instance was restored without an LLM client. "
            "Provide one when calling from_json(..., llm_client=your_client) "
            "if you need to run the algorithm."
        )


class Experiment(ABC):
    """Abstract experiment runner coordinating iteration objects."""

    algo_label: str = "Experiment"

    # Validation requirements - override in subclasses as needed
    # Base class has no requirements (it's abstract); subclasses set their own
    _require_solutions: bool = False
    _require_metrics: bool = False
    _require_llm: bool = False

    def __init__(self, algo_config: Dict[str, Any]) -> None:
        self.algo_config = dict(algo_config or {})
        self.solutions_df = self._coerce_dataframe(self.algo_config.get("solutions_df"))
        self.metric_columns = list(self.algo_config.get("metric_columns") or [])
        self.non_numeric_metrics = set(self.algo_config.get("non_numeric_metrics") or [])
        self.numeric_metrics = [c for c in self.metric_columns if c not in self.non_numeric_metrics]
        self.llm_client = self.algo_config.get("llm_client")
        self.prompt_template = self.algo_config.get("prompt_template")
        self.iterations = int(self.algo_config.get("iterations", 3))
        self.seed = self.algo_config.get("seed")
        self.llm_params = dict(self.algo_config.get("llm_params") or {})
        self.gtu_weights: Dict[str, Any] = dict(self.algo_config.get("gtu_weights") or {})
        self.metric_signs: Dict[str, int] = dict(self.algo_config.get("metric_signs") or {})
        self.human_sol = list(self.algo_config.get("human_sol") or [])
        # Mutable shared state
        self._history: Optional[Dict[str, Any]] = None
        self._plot_data: Optional[Dict[str, Any]] = None
        self._winner_idx: Optional[int] = None
        self._winner_solution: Optional[Dict[str, Any]] = None
        self.prompt_history: list = []
        self._nar: int = 0
        self._metadata: Dict[str, Any] = {}

        # Validate requirements based on class attributes
        self._validate_common_requirements()

    # ---- Validation -------------------------------------------------------

    def _validate_common_requirements(self) -> None:
        """Validate common requirements based on class attributes."""
        algo_name = getattr(self, "algo_label", self.__class__.__name__)
        if self._require_solutions and (self.solutions_df is None or self.solutions_df.empty):
            raise ValueError(f"solutions_df is required for {algo_name}.")
        if self._require_metrics and not self.metric_columns:
            raise ValueError(f"metric_columns is required for {algo_name}.")
        if self._require_llm and self.llm_client is None:
            raise ValueError(f"llm_client is required for {algo_name}.")

    # ---- Template methods -------------------------------------------------

    def make_iteration(self, i: int, total: int, ctx: Dict[str, Any]) -> AlgorithmIteration:
        """Return an AlgorithmIteration for iteration i (override in subclasses)."""
        raise NotImplementedError("make_iteration must be implemented by subclasses")

    def _build_history(
        self,
        records: list[IterationResult],
        start_time: str,
        end_time: str,
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construct history payload for all iterations so far."""
        history: Dict[str, Any] = {
            "metadata": dict(self._metadata),
            "batch_comparisons": [],
        }

        for idx, rec in enumerate(records, start=1):
            history["batch_comparisons"].append(
                {
                    "batch_num": idx,
                    "winner_idx": rec.winner_idx,
                    "winner_solution": rec.winner_solution,
                    "prompt": rec.prompt,
                    "response": rec.response,
                }
            )
        return history

    def save_metadata(self, start_time: str, end_time: Optional[str], ctx: Dict[str, Any]) -> None:
        """Persist metadata once per run."""
        n_solutions = len(self.solutions_df) if self.solutions_df is not None else 0
        metadata: Dict[str, Any] = {
            "iterations": self.iterations,
            "n_solutions": n_solutions,
            "metric_columns": self.metric_columns,
            "start_time": start_time,
            "end_time": end_time,
            "algorithm": getattr(self, "algo_label", self.__class__.__name__),
        }
        batch_size = getattr(self, "batch_size", None)
        if batch_size is not None:
            metadata["batch_size"] = batch_size
        self._metadata = metadata

    # ---- Public run -------------------------------------------------------

    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        start_time = datetime.now().isoformat()
        ctx: Dict[str, Any] = {}
        records: list[IterationResult] = []
        self.prompt_history = []
        self._nar = 0
        self.save_metadata(start_time, None, ctx)

        for i in range(self.iterations):
            iteration_obj = self.make_iteration(i, self.iterations, ctx)
            result = iteration_obj.execute()
            records.append(result)
            self._nar += result.nar_delta
            if result.prompt or result.response:
                self.prompt_history.append(
                    {"prompt": result.prompt, "response": result.response, "stage": result.stage}
                )
            # update winner when result deemed final or on last iteration
            if result.stage in {"final", "single", "utility", "full_batch"} or i == self.iterations - 1:
                self._winner_idx = result.winner_idx
                self._winner_solution = result.winner_solution

            # progressively update history so intermediate state is available
            current_end = datetime.now().isoformat()
            self._history = self._build_history(records, start_time, current_end, ctx)

        end_time = datetime.now().isoformat()
        self.save_metadata(start_time, end_time, ctx)
        self._history = self._build_history(records, start_time, end_time, ctx)
        self._plot_data = None

        return {
            "final_winner_idx": self._winner_idx,
            "final_winner_solution": self._winner_solution,
            "history": self._history,
            "plot_data": self._plot_data,
            "prompt_history": self.prompt_history,
        }

    # ---- Getters ----------------------------------------------------------

    def get_winner(self) -> Dict[str, Any]:
        return self._winner_solution

    def get_ith_winner(self, i: int) -> Dict[str, Any]:
        """Get the winner solution for batch i (1-indexed)."""
        for comp in self._history["batch_comparisons"]:
            if comp["batch_num"] == i:
                return comp["winner_solution"]
        raise KeyError(f"No batch with batch_num={i}")

    def get_ith_winner_idx(self, i: int) -> int:
        """Get the winner index for batch i (1-indexed)."""
        for comp in self._history["batch_comparisons"]:
            if comp["batch_num"] == i:
                return comp["winner_idx"]
        raise KeyError(f"No batch with batch_num={i}")

    def get_all_winner_indices(self, up_to_batch: Optional[int] = None) -> list[int]:
        """Get all winner indices from history, optionally up to a specific batch number."""
        winners = []
        for comp in self._history["batch_comparisons"]:
            if up_to_batch is not None and comp["batch_num"] > up_to_batch:
                break
            winners.append(comp["winner_idx"])
        return winners

    def get_rank(self, idx: int) -> float:
        return self._lookup_rank(idx)

    def get_nar(self) -> float:
        rank = self._lookup_rank(self._winner_idx)
        n = len(self.solutions_df)
        return rank / n

    def get_gtu(self, idx: int) -> float:
        solution = self.solutions_df.iloc[idx]
        total = 0.0
        for k in self.numeric_metrics:
            if k in self.gtu_weights:
                #print(f"Applying weight {self.gtu_weights[k]} to metric {k} with value {solution[k]}")
                total += float(self.gtu_weights[k]) * float(solution[k])
        return total

    def get_scenario(self) -> str:
        return self.algo_config["scenario"]

    def get_mode(self) -> str:
        return self.algo_config["mode"]

    def get_algo(self) -> str:
        return self.algo_label

    def get_prompt(self) -> str:
        return self.prompt_history[-1]["prompt"]

    def get_llm_response(self) -> str:
        return self.prompt_history[-1]["response"]

    # ---- Serialization ----------------------------------------------------

    def to_json(self) -> Dict[str, Any]:
        config = {
            "iterations": self.iterations,
            "metric_columns": self.metric_columns,
            "seed": self.seed,
            "llm_params": self.llm_params,
            "gtu_weights": self.gtu_weights,
            "metric_signs": self.metric_signs,
            "human_sol": self.human_sol,
            "llm_client_present": self.llm_client is not None,
            "prompt_template_class": getattr(self.prompt_template, "__class__", type("X", (), {})).__name__ if self.prompt_template else None,
        }
        winner_solution = (
            self._winner_solution.to_dict()  # type: ignore[attr-defined]
            if hasattr(self._winner_solution, "to_dict")
            else self._winner_solution
        )
        return {
            "algo": self.get_algo(),
            "config": config,
            "solutions": self.solutions_df.to_dict(orient="records") if self.solutions_df is not None else None,
            "history": self._history,
            "plot_data": self._plot_data,
            "winner_idx": self._winner_idx,
            "winner_solution": winner_solution,
            "nar": self._nar,
            "prompt_history": self.prompt_history,
        }

    @staticmethod
    def from_json(source: "str | Path | Dict[str, Any]") -> "Experiment":
        import json
        if isinstance(source, (str, Path)):
            with open(source, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = source

        algo_name = (
            data.get("algo")
            or (data.get("meta") or {}).get("algo")
            or "Experiment"
        ).lower()

        if "tournament" in algo_name:
            from tournament_algorithm import TournamentExperiment
            cls = TournamentExperiment
        elif "utility" in algo_name:
            from utility_algorithm import UtilityExperiment
            cls = UtilityExperiment
        elif "baseline" in algo_name:
            from baseline_algorithm import BaselineExperiment
            cls = BaselineExperiment
        elif "full_batch" in algo_name or "fullbatch" in algo_name:
            from full_batch_algorithm import FullBatchExperiment
            cls = FullBatchExperiment
        else:
            cls = Experiment
        return cls._from_json_data(data)

    @classmethod
    def _from_json_data(cls, data: Dict[str, Any]) -> "Experiment":
        meta = data.get("meta") or {}
        raw_config = data.get("config") or meta.get("config") or {}
        config = dict(raw_config)
        for key in ("scenario", "tag", "mode", "api_model", "model_name"):
            if key not in config and meta.get(key):
                config[key] = meta[key]

        # Pull mode-specific keys (e.g. human_sol, weights) into top-level config
        mode = config.get("mode")
        if mode:
            mode_block = (config.get("modes") or {}).get(mode) or {}
            for key in ("human_sol", "weights"):
                if key not in config and key in mode_block:
                    config[key] = mode_block[key]

        solutions = data.get("solutions") or (data.get("results") or {}).get("solutions")
        if solutions is not None:
            config["solutions_df"] = pd.DataFrame(solutions)
        elif config.get("data_csv"):
            root = Path(__file__).resolve().parent
            data_path = root / config["data_csv"]
            df = pd.read_csv(data_path)
            if config.get("metric_columns"):
                df = df.dropna(subset=config["metric_columns"])
            config["solutions_df"] = df


        config["llm_client"] = _NullLLMClient()

        algo = cls(config)
        state = data.get("results") or data
        algo._history = state.get("history")
        algo._plot_data = state.get("plot_data")
        algo._winner_idx = state.get("winner_idx") or state.get("final_winner_idx")
        algo._winner_solution = state.get("winner_solution") or state.get("final_winner_solution")
        algo._nar = int(state.get("nar", 0) or 0)
        algo.prompt_history = state.get("prompt_history") or []

        algo.gtu_weights = dict(config.get("gtu_weights") or {})
        algo.metric_signs = dict(config.get("metric_signs") or {})
        algo.human_sol = list(config.get("human_sol") or [])
        if not algo.human_sol and algo.solutions_df is not None:
            algo.human_sol = list(range(len(algo.solutions_df)))
        return algo

    # ---- Helpers ----------------------------------------------------------

    @staticmethod
    def _coerce_dataframe(df_like: Any) -> Optional[pd.DataFrame]:
        if df_like is None:
            return None
        if isinstance(df_like, pd.DataFrame):
            return df_like.reset_index(drop=True)
        try:
            return pd.DataFrame(df_like)
        except Exception:
            return None

    def _lookup_rank(self, idx: int) -> float:
        if idx in self.human_sol:
            return self.human_sol.index(idx) + 1
        m = len(self.human_sol)
        N = len(self.solutions_df)
        return (m + 1 + N) / 2
