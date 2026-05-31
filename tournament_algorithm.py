from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from experiment import Experiment
from iteration import AlgorithmIteration, IterationResult
from prompt_tournament import ComparisonPromptAdapter, _excel_column_label


class TournamentIteration(AlgorithmIteration):
    """Handles prelim and final stages based on iteration index."""

    def get_batch_indices(self, batch_num: int) -> List[int]:
        """
        Convenience helper to fetch candidate indices for a given batch number
        from the experiment's stored history.
        """
        history = self.experiment._history or {}
        comps = (history.get("batch_comparisons") or [])
        for comp in comps:
            if int(comp.get("batch_num") or 0) == int(batch_num):
                return list(comp.get("batch_indices") or [])
        raise ValueError(f"Batch {batch_num} not found in history")

    def __init__(
        self,
        experiment,
        iteration_index: int,
        total_iterations: int,
        ctx: Dict[str, Any],
        batch_size: int,
    ):
        super().__init__(experiment, iteration_index, total_iterations, ctx)
        self.batch_size = batch_size

    def _sample_indices(self, n: int) -> List[int]:
        total = len(self.experiment.solutions_df)
        human_sol = self.experiment.human_sol or []
        if self.experiment.unique_rank_batch and human_sol:
            # Guarantee at least one human_sol index in the batch
            valid_human = [i for i in human_sol if i < total]
            non_human = [i for i in range(total) if i not in set(valid_human)]
            picked_human = random.sample(valid_human, min(1, len(valid_human)))
            remaining = n - len(picked_human)
            pool = [i for i in non_human if i not in set(picked_human)]
            rest = random.sample(pool, min(remaining, len(pool)))
            batch = picked_human + rest
            random.shuffle(batch)
            return batch
        idxs = list(range(total))
        random.shuffle(idxs)
        return idxs[:n]

    def _present_and_choose(self, candidate_indices: List[int]) -> Tuple[Optional[int], Optional[str], str, str, Optional[Dict]]:
        items: List[Dict[str, Any]] = [self.experiment.solutions_df.iloc[i].to_dict() for i in candidate_indices]
        prompt = self.experiment.prompt_template.format(items)
        response = self.experiment.llm_client.generate_response(prompt, **self.experiment.llm_params)

        # Grab logprobs if the client supports them
        logprobs = getattr(self.experiment.llm_client, "_last_logprobs", None)

        try:
            idx = self.experiment.prompt_template.parse_response(response, num_options=len(candidate_indices))
        except Exception:
            idx = None
        chosen_idx = idx

        winner: Optional[int] = None
        if idx is not None and 0 <= idx < len(candidate_indices):
            winner = candidate_indices[idx]

        letter: Optional[str]
        if chosen_idx is not None and isinstance(chosen_idx, int) and 0 <= chosen_idx < len(candidate_indices):
            letter = _excel_column_label(chosen_idx)
        else:
            letter = None

        return winner, letter, prompt, response, logprobs

    def execute(self) -> IterationResult:
        total = self.total_iterations
        is_single = total == 1
        is_final = not is_single and self.iteration_index == total - 1

        if is_single:
            stage = "single"
            candidates = self._sample_indices(self.batch_size)
        elif is_final:
            stage = "final"
            # Use experiment's history to get all winner indices from prelim rounds
            # iteration_index is 0-indexed, batch_num is 1-indexed
            # So up_to_batch=iteration_index gets winners from batch 1 to iteration_index
            champions = self.experiment.get_all_winner_indices(up_to_batch=self.iteration_index)
            candidates = champions if champions else self._sample_indices(self.batch_size)
        else:
            stage = "prelim"
            candidates = self._sample_indices(self.batch_size)

        winner_idx, letter, prompt, response, logprobs = self._present_and_choose(candidates)
        winner_solution = (
            self.experiment.solutions_df.iloc[winner_idx].to_dict()
            if winner_idx is not None
            else None
        )

        algo_specific = {
            "batch_indices": candidates,
            "choice_letter": letter,
            "stage": stage,
        }
        if logprobs is not None:
            algo_specific["logprobs"] = logprobs

        return IterationResult(
            winner_idx=winner_idx,
            winner_solution=winner_solution,
            stage=stage,
            prompt=prompt,
            response=response,
            nar_delta=1,
            algo_specific=algo_specific,
        )


class TournamentExperiment(Experiment):
    """LISTEN-T implemented with iterations."""

    algo_label = "TournamentExperiment"
    _require_solutions = True
    _require_metrics = True

    def __init__(self, algo_config: Dict[str, Any]) -> None:
        super().__init__(algo_config)
        bs_cfg = self.algo_config.get("batch_size")
        self.batch_size = int(bs_cfg) if bs_cfg is not None else int(self.algo_config.get("tournament_batch_size", 50))
        self.unique_rank_batch = bool(self.algo_config.get("unique_rank_batch", False))
        assert self.iterations == 1 or self.iterations >= 3, "iterations must be 1 or >= 3"
        if self.seed is not None:
            random.seed(int(self.seed))

    def make_iteration(self, i: int, total: int, ctx: Dict[str, Any]) -> AlgorithmIteration:
        return TournamentIteration(
            self,
            i,
            total,
            ctx,
            batch_size=self.batch_size,
        )

    def _build_history(self, records: List[IterationResult], start_time: str, end_time: str, ctx: Dict[str, Any]):
        history = super()._build_history(records, start_time, end_time, ctx)
        history.setdefault("metadata", {})["tournament_spec"] = "LISTEN-T"
        for entry, rec in zip(history.get("batch_comparisons", []), records):
            entry["stage"] = rec.stage
            algo_specific = rec.algo_specific or {}
            if "batch_indices" in algo_specific:
                entry["batch_indices"] = algo_specific["batch_indices"]
            if "choice_letter" in algo_specific:
                entry["choice_letter"] = algo_specific["choice_letter"]
            if "logprobs" in algo_specific:
                entry["logprobs"] = algo_specific["logprobs"]
        return history

    def to_json(self) -> Dict[str, Any]:
        payload = super().to_json()
        payload["config"]["batch_size"] = self.batch_size
        payload["config"]["unique_rank_batch"] = self.unique_rank_batch
        return payload


__all__ = ["TournamentExperiment"]
