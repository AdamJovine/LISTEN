from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class IterationResult:
    """Normalized record returned by each AlgorithmIteration."""

    winner_idx: Optional[int]
    winner_solution: Any
    stage: str
    prompt: Optional[str] = None
    response: Optional[str] = None
    nar_delta: int = 0
    algo_specific: Optional[Dict[str, Any]] = None


class AlgorithmIteration:
    """Base class for a single iteration of an experiment.

    Subclasses implement `execute` and may read/write shared context.
    """

    def __init__(self, experiment: Any, iteration_index: int, total_iterations: int, ctx: Dict[str, Any]):
        self.experiment = experiment
        self.iteration_index = iteration_index
        self.total_iterations = total_iterations
        # shared mutable dict provided by Experiment.run
        self.ctx = ctx

    def execute(self) -> IterationResult:  # pragma: no cover - interface
        raise NotImplementedError
