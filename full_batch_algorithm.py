from __future__ import annotations

import random
from typing import Any, Dict, List

from tournament_algorithm import TournamentExperiment
from iteration import AlgorithmIteration, IterationResult
from prompt_tournament import _excel_column_label


# Conservative chars-per-token for JSON-heavy prompts.
# JSON with numbers/punctuation tokenizes at ~3 chars/token on llama tokenizers.
_CHARS_PER_TOKEN = 3

# Default context window limits (tokens) by provider.
_CONTEXT_LIMITS = {
    "groq": 128_000,
    "gemini": 1_000_000,
}
_OUTPUT_RESERVE = 8_192   # tokens reserved for LLM output
_SAFETY_FACTOR = 0.90     # use only 90% of estimated budget to avoid edge cases


class FullBatchIteration(AlgorithmIteration):
    """Single full-batch iteration (always one)."""

    def execute(self) -> IterationResult:
        exp = self.experiment
        if exp.llm_client is None or exp.prompt_template is None:
            raise ValueError("llm_client and prompt_template are required.")

        all_indices = list(range(len(exp.solutions_df)))
        random.shuffle(all_indices)

        batch_size = getattr(exp, "batch_size", None) or len(all_indices)
        candidate_indices = all_indices[:batch_size]

        items = [exp.solutions_df.iloc[i].to_dict() for i in candidate_indices]
        prompt = exp.prompt_template.format(items)
        response = exp.llm_client.generate_response(prompt, **exp.llm_params)
        print(f"[FullBatchExperiment] Using batch_size={len(candidate_indices)}")

        try:
            idx = exp.prompt_template.parse_response(response, num_options=len(candidate_indices))
        except Exception:
            idx = None

        if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(candidate_indices):
            letter = None
            winner_idx = None
            winner_solution = None
        else:
            letter = _excel_column_label(idx)
            winner_idx = candidate_indices[idx]
            winner_solution = exp.solutions_df.iloc[winner_idx].to_dict()

        algo_specific = {
            "batch_indices": all_indices,
            "choice_letter": letter,
            "presentation_order": all_indices,
        }

        return IterationResult(
            winner_idx=winner_idx,
            winner_solution=winner_solution,
            stage="full_batch",
            prompt=prompt,
            response=response,
            nar_delta=1,
            algo_specific=algo_specific,
        )


class FullBatchExperiment(TournamentExperiment):
    """Full Batch method built on TournamentExperiment base.

    Feeds as many solutions as possible into a single LLM call.
    If all solutions exceed the context window, automatically caps
    the batch size to the maximum that fits.
    """

    algo_label = "FullBatchExperiment"

    def __init__(self, algo_config: Dict[str, Any]) -> None:
        super().__init__(algo_config)
        self.iterations = 1
        self.scenario = algo_config.get("scenario", "unknown")

        if self.solutions_df is not None and self.prompt_template is not None:
            explicit_batch = algo_config.get("batch_size")
            n_solutions = len(self.solutions_df)
            max_fitting = self._max_batch_for_context(algo_config)

            if explicit_batch is not None:
                self.batch_size = min(int(explicit_batch), max_fitting)
            else:
                self.batch_size = min(n_solutions, max_fitting)

            if self.batch_size < n_solutions:
                print(f"[FullBatchExperiment] Capped batch_size to {self.batch_size} "
                      f"(of {n_solutions}) to fit context window")

    def _max_batch_for_context(self, algo_config: Dict[str, Any]) -> int:
        """Find the max batch size that fits by building the actual prompt.

        Uses binary search: formats the real prompt at candidate sizes
        and checks against the token budget.
        """
        provider = algo_config.get("api_model", "groq")
        context_limit = _CONTEXT_LIMITS.get(provider, 128_000)
        max_input_tokens = int((context_limit - _OUTPUT_RESERVE) * _SAFETY_FACTOR)

        n = len(self.solutions_df)
        items_all = [self.solutions_df.iloc[i].to_dict() for i in range(min(n, 5000))]

        def _prompt_tokens(size: int) -> int:
            prompt = self.prompt_template.format(items_all[:size])
            return len(prompt) // _CHARS_PER_TOKEN

        # Binary search for the largest size that fits
        lo, hi = 1, len(items_all)
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _prompt_tokens(mid) <= max_input_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return best

    def make_iteration(self, i: int, total: int, ctx: Dict[str, Any]) -> AlgorithmIteration:
        return FullBatchIteration(self, i, total, ctx)

    def _build_history(self, records: List[IterationResult], start_time: str, end_time: str, ctx: Dict[str, Any]):
        rec = records[0] if records else None
        history = {
            "metadata": {
                "batch_size": len((rec.algo_specific or {}).get("batch_indices", [])) if rec else 0,
                "original_n_solutions": len(self.solutions_df),
                "iterations": 1,
                "n_solutions": len(self.solutions_df),
                "metric_columns": self.metric_columns,
                "start_time": start_time,
                "tournament_spec": "full_batch",
                "end_time": end_time,
            },
            "batch_comparisons": [],
        }
        if rec:
            history["batch_comparisons"].append(
                {
                    "batch_num": 1,
                    "stage": "full_batch",
                    "batch_indices": (rec.algo_specific or {}).get("batch_indices"),
                    "winner_idx": rec.winner_idx,
                    "choice_letter": (rec.algo_specific or {}).get("choice_letter"),
                    "winner_solution": rec.winner_solution,
                    "prompt": rec.prompt,
                    "response": rec.response,
                    "presentation_order": (rec.algo_specific or {}).get("presentation_order"),
                }
            )
        return history


# Keep old name as alias for backward compatibility
FullBatchAlgorithm = FullBatchExperiment

__all__ = ["FullBatchExperiment", "FullBatchAlgorithm"]
