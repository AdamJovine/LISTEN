"""Tests for to_json and from_json methods of TournamentExperiment and UtilityExperiment."""

from pathlib import Path
import sys

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment import Experiment
from tournament_algorithm import TournamentExperiment
from utility_algorithm import UtilityExperiment


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    def generate_response(self, prompt, **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return "A"


class MockPromptTemplate:
    """Mock prompt template for testing."""

    def format(self, items_or_iteration, **kwargs):
        return f"Mock prompt for {len(items_or_iteration) if isinstance(items_or_iteration, list) else items_or_iteration} items"

    def parse_response(self, response, num_options=None):
        # Return index 0 for simple letter responses
        if response and response[0].isalpha():
            return ord(response[0].upper()) - ord('A')
        return 0


# =============================================================================
# TournamentExperiment Tests
# =============================================================================


def test_tournament_to_json_includes_batch_size():
    """TournamentExperiment should serialize batch_size in config."""
    df = pd.DataFrame([
        {"score": 1.0, "quality": 2.0},
        {"score": 3.0, "quality": 1.0},
        {"score": 2.0, "quality": 3.0},
    ])
    config = {
        "solutions_df": df,
        "metric_columns": ["score", "quality"],
        "batch_size": 2,
        "iterations": 3,
        "seed": 42,
    }
    algo = TournamentExperiment(config)

    payload = algo.to_json()

    assert payload["algo"] == "TournamentExperiment"
    assert algo.batch_size == 2
    # batch_size should be serialized in the JSON config
    assert payload["config"]["batch_size"] == 2


def test_tournament_to_json_serializes_history_structure():
    """TournamentExperiment history should have batch_comparisons structure."""
    df = pd.DataFrame([
        {"score": 1.0, "quality": 2.0},
        {"score": 3.0, "quality": 1.0},
        {"score": 2.0, "quality": 3.0},
    ])
    config = {
        "solutions_df": df,
        "metric_columns": ["score", "quality"],
        "batch_size": 2,
        "iterations": 3,
        "seed": 42,
        "llm_client": MockLLMClient(["A", "B", "A"]),
        "prompt_template": MockPromptTemplate(),
    }
    algo = TournamentExperiment(config)
    algo.run()

    payload = algo.to_json()

    assert "history" in payload
    assert "batch_comparisons" in payload["history"]
    assert len(payload["history"]["batch_comparisons"]) == 3  # 2 prelim + 1 final

    # Check structure of batch_comparisons
    for comp in payload["history"]["batch_comparisons"]:
        assert "batch_num" in comp
        assert "stage" in comp
        assert "batch_indices" in comp
        assert "winner_idx" in comp
        assert "choice_letter" in comp
        assert "winner_solution" in comp


def test_tournament_from_json_restores_history():
    """Experiment.from_json should restore history with batch_comparisons."""
    solutions = [
        {"score": 1.0, "quality": 2.0},
        {"score": 3.0, "quality": 1.0},
    ]
    batch_comparisons = [
        {
            "batch_num": 1,
            "stage": "prelim",
            "batch_indices": [0, 1],
            "winner_idx": 1,
            "choice_letter": "B",
            "winner_solution": solutions[1],
        },
        {
            "batch_num": 2,
            "stage": "final",
            "batch_indices": [1],
            "winner_idx": 1,
            "choice_letter": "A",
            "winner_solution": solutions[1],
        },
    ]
    data = {
        "algo": "TournamentExperiment",
        "config": {
            "metric_columns": ["score", "quality"],
            "batch_size": 2,
            "iterations": 3,
        },
        "solutions": solutions,
        "history": {
            "metadata": {"batch_size": 2, "iterations": 3},
            "batch_comparisons": batch_comparisons,
        },
        "winner_idx": 1,
        "winner_solution": solutions[1],
        "nar": 2,
        "prompt_history": [],
    }

    algo = Experiment.from_json(data)

    assert algo._history is not None
    assert "batch_comparisons" in algo._history
    assert len(algo._history["batch_comparisons"]) == 2
    assert algo._winner_idx == 1
    assert algo._winner_solution == solutions[1]
    assert algo.batch_size == 2


def test_tournament_round_trip_preserves_state():
    """to_json followed by from_json should preserve TournamentExperiment state."""
    df = pd.DataFrame([
        {"score": 1.0, "quality": 2.0},
        {"score": 3.0, "quality": 1.0},
        {"score": 2.0, "quality": 3.0},
    ])
    config = {
        "solutions_df": df,
        "metric_columns": ["score", "quality"],
        "batch_size": 2,
        "iterations": 3,
        "seed": 42,
        "llm_client": MockLLMClient(["A", "B", "A"]),
        "prompt_template": MockPromptTemplate(),
    }
    algo = TournamentExperiment(config)
    algo.run()

    payload = algo.to_json()
    restored = Experiment.from_json(payload)

    assert restored._winner_idx == algo._winner_idx
    assert restored._winner_solution == algo._winner_solution
    assert restored._nar == algo._nar
    assert len(restored._history["batch_comparisons"]) == len(algo._history["batch_comparisons"])


def test_tournament_from_json_with_nested_results():
    """Experiment.from_json should handle results nested under 'results' key."""
    solutions = [{"score": 1.0}, {"score": 2.0}]
    data = {
        "config": {
            "metric_columns": ["score"],
            "batch_size": 2,
            "iterations": 3,
        },
        "solutions": solutions,
        "results": {
            "history": {"batch_comparisons": []},
            "final_winner_idx": 0,
            "final_winner_solution": solutions[0],
            "nar": 1,
        },
    }

    algo = Experiment.from_json(data)

    assert algo._winner_idx == 0
    assert algo._winner_solution == solutions[0]


# =============================================================================
# UtilityExperiment Tests
# =============================================================================


class MockUtilityPromptTemplate:
    """Mock prompt template for UtilityExperiment testing."""

    def format(self, iteration=0, **kwargs):
        return f"Mock utility prompt for iteration {iteration}"

    def parse_response(self, response):
        return {
            "weights": {"score": 1.0, "quality": -0.5},
            "formula": "utility = sum(weight_i * score_i)",
            "description": "Test weights",
        }


def test_utility_to_json_serializes_weights():
    """UtilityExperiment should serialize LLM-produced weights."""
    df = pd.DataFrame([
        {"score": 1.0, "quality": 2.0},
        {"score": 3.0, "quality": 1.0},
    ])
    mock_response = '{"weights": {"score": 1.0, "quality": -0.5}, "formula": "test", "description": "test"}'
    config = {
        "solutions_df": df,
        "metric_columns": ["score", "quality"],
        "iterations": 1,
        "seed": 42,
        "llm_client": MockLLMClient([mock_response]),
        "prompt_template": MockUtilityPromptTemplate(),
    }
    algo = UtilityExperiment(config)
    algo.run()

    # Check that weights were tracked on the object
    assert len(algo.weights) == 1
    assert "score" in algo.weights[0].index
    assert "quality" in algo.weights[0].index

    # Check that weights are serialized in JSON payload
    payload = algo.to_json()
    assert "weights" in payload
    assert len(payload["weights"]) == 1
    assert payload["weights"][0]["score"] == 1.0
    assert payload["weights"][0]["quality"] == -0.5


def test_utility_to_json_serializes_iteration_history():
    """UtilityExperiment history should have iterations structure."""
    df = pd.DataFrame([
        {"score": 1.0, "quality": 2.0},
        {"score": 3.0, "quality": 1.0},
    ])
    mock_response = '{"weights": {"score": 1.0, "quality": -0.5}, "formula": "test", "description": "test"}'
    config = {
        "solutions_df": df,
        "metric_columns": ["score", "quality"],
        "iterations": 2,
        "seed": 42,
        "llm_client": MockLLMClient([mock_response, mock_response]),
        "prompt_template": MockUtilityPromptTemplate(),
    }
    algo = UtilityExperiment(config)
    algo.run()

    assert algo._history is not None
    assert "iterations" in algo._history
    assert len(algo._history["iterations"]) == 2

    for iteration in algo._history["iterations"]:
        assert "iteration" in iteration
        assert "utility" in iteration
        assert "weights" in iteration
        assert "best_solution" in iteration


def test_utility_from_json_restores_state():
    """Experiment.from_json should restore algorithm state."""
    solutions = [
        {"score": 1.0, "quality": 2.0},
        {"score": 3.0, "quality": 1.0},
    ]
    data = {
        "algo": "UtilityExperiment",
        "config": {
            "metric_columns": ["score", "quality"],
            "iterations": 2,
        },
        "solutions": solutions,
        "history": {
            "iterations": [
                {"iteration": 1, "utility": 0.5, "winner_idx": 0},
                {"iteration": 2, "utility": 0.8, "winner_idx": 1},
            ]
        },
        "winner_idx": 1,
        "winner_solution": solutions[1],
        "nar": 2,
        "prompt_history": ["prompt1", "prompt2"],
    }

    algo = Experiment.from_json(data)

    assert algo._history is not None
    assert "iterations" in algo._history
    assert len(algo._history["iterations"]) == 2
    assert algo._winner_idx == 1
    assert algo._winner_solution == solutions[1]
    assert algo.prompt_history == ["prompt1", "prompt2"]


def test_utility_from_json_handles_non_numeric_metrics():
    """Experiment.from_json should restore non_numeric_metrics from config."""
    solutions = [
        {"score": 1.0, "name": "Product A"},
        {"score": 2.0, "name": "Product B"},
    ]
    data = {
        "algo": "UtilityExperiment",
        "config": {
            "metric_columns": ["score", "name"],
            "non_numeric_metrics": ["name"],
            "iterations": 1,
        },
        "solutions": solutions,
    }

    algo = Experiment.from_json(data)

    assert algo.metric_columns == ["score", "name"]
    assert algo.non_numeric_metrics == {"name"}
    assert algo.numeric_metrics == ["score"]


def test_utility_round_trip_preserves_state():
    """to_json followed by from_json should preserve UtilityExperiment state."""
    df = pd.DataFrame([
        {"score": 1.0, "quality": 2.0},
        {"score": 3.0, "quality": 1.0},
    ])
    mock_response = '{"weights": {"score": 1.0, "quality": -0.5}, "formula": "test", "description": "test"}'
    config = {
        "solutions_df": df,
        "metric_columns": ["score", "quality"],
        "iterations": 1,
        "seed": 42,
        "llm_client": MockLLMClient([mock_response]),
        "prompt_template": MockUtilityPromptTemplate(),
    }
    algo = UtilityExperiment(config)
    algo.run()

    payload = algo.to_json()
    restored = Experiment.from_json(payload)

    assert restored._winner_idx == algo._winner_idx
    assert restored._nar == algo._nar
    assert len(restored._history["iterations"]) == len(algo._history["iterations"])


# =============================================================================
# Cross-algorithm Tests
# =============================================================================


def test_from_json_infers_tournament_algorithm():
    """Experiment.from_json should return TournamentExperiment for tournament data."""
    from experiment import Experiment

    data = {
        "algo": "TournamentExperiment",
        "meta": {
            "scenario": "exam",
            "mode": "REGISTRAR",
            "api_model": "groq",
        },
        "config": {"metric_columns": ["score"], "batch_size": 4, "iterations": 3},
        "solutions": [{"score": 1.0}, {"score": 2.0}],
        "history": {"batch_comparisons": []},
        "winner_idx": 1,
    }

    algo = Experiment.from_json(data)

    assert isinstance(algo, TournamentExperiment)
    assert algo.get_algo() == "TournamentExperiment"
    assert algo.get_scenario() == "exam"
    assert algo.get_mode() == "REGISTRAR"
    assert algo.batch_size == 4


def test_from_json_infers_utility_algorithm():
    """Experiment.from_json should return UtilityExperiment for utility data."""
    from experiment import Experiment

    data = {
        "algo": "UtilityExperiment",
        "meta": {
            "scenario": "headphones",
            "mode": "SOFT",
        },
        "config": {"metric_columns": ["score"], "iterations": 1},
        "solutions": [{"score": 1.0}, {"score": 2.0}],
        "history": {"iterations": []},
        "winner_idx": 1,
    }

    algo = Experiment.from_json(data)

    assert isinstance(algo, UtilityExperiment)
    assert algo.get_algo() == "UtilityExperiment"
    assert algo.get_scenario() == "headphones"
    assert algo.get_mode() == "SOFT"


def test_from_json_reads_algo_from_meta():
    """Experiment.from_json should find algo type in meta if not at top level."""
    from experiment import Experiment

    data = {
        "meta": {
            "algo": "tournament",
            "scenario": "flights_chi_nyc",
            "mode": "Complicated",
        },
        "config": {"metric_columns": ["cost"], "batch_size": 2, "iterations": 3},
        "solutions": [{"cost": 100}, {"cost": 200}],
        "winner_idx": 1,
    }

    algo = Experiment.from_json(data)

    assert isinstance(algo, TournamentExperiment)
    assert algo.get_scenario() == "flights_chi_nyc"
    assert algo.get_mode() == "Complicated"


def test_from_json_copies_metadata_to_config():
    """Metadata from 'meta' should be accessible via getters."""
    from experiment import Experiment

    data = {
        "algo": "TournamentExperiment",
        "meta": {
            "scenario": "exam",
            "mode": "BASE",
            "api_model": "groq",
            "model_name": "llama-3.3-70b-versatile",
        },
        "config": {"metric_columns": ["score"], "batch_size": 2, "iterations": 3},
        "solutions": [{"score": 1.0}],
        "winner_idx": 1,
    }

    algo = Experiment.from_json(data)

    assert algo.get_scenario() == "exam"
    assert algo.get_mode() == "BASE"
    assert algo.algo_config.get("api_model") == "groq"
    assert algo.algo_config.get("model_name") == "llama-3.3-70b-versatile"


def test_from_json_distinguishes_algorithm_types():
    """from_json should work correctly for both algorithm types."""
    solutions = [{"score": 1.0}, {"score": 2.0}, {"score": 3.0}]

    tournament_data = {
        "algo": "TournamentExperiment",
        "config": {"metric_columns": ["score"], "batch_size": 2, "iterations": 3},
        "solutions": solutions,
        "history": {"batch_comparisons": []},
        "winner_idx": 2,  # Use non-zero to avoid falsy issue in base class
    }

    utility_data = {
        "algo": "UtilityExperiment",
        "config": {"metric_columns": ["score"], "iterations": 1},
        "solutions": solutions,
        "history": {"iterations": []},
        "winner_idx": 1,
    }

    tournament_algo = Experiment.from_json(tournament_data)
    utility_algo = Experiment.from_json(utility_data)

    assert isinstance(tournament_algo, TournamentExperiment)
    assert isinstance(utility_algo, UtilityExperiment)
    assert tournament_algo._winner_idx == 2
    assert utility_algo._winner_idx == 1
