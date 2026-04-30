import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utility_algorithm import UtilityExperiment


class DummyLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def generate_response(self, prompt, **kwargs):
        resp = self.responses[self.call_count] if self.call_count < len(self.responses) else self.responses[-1]
        self.call_count += 1
        return resp


class DummyPrompt:
    def format(self, iteration=0, **kwargs):
        return f"iter {iteration}"

    def parse_response(self, response):
        return {"weights": {"score": 1.0, "quality": -0.5}, "formula": "utility = sum()", "description": "dummy"}


def make_algo():
    df = pd.DataFrame([{"score": 1.0, "quality": 2.0}, {"score": 3.0, "quality": 1.0}])
    mock_response = '{"weights": {"score": 1.0, "quality": -0.5}, "formula": "test", "description": "test"}'
    config = {
        "solutions_df": df,
        "metric_columns": ["score", "quality"],
        "iterations": 1,
        "llm_client": DummyLLM([mock_response]),
        "prompt_template": DummyPrompt(),
    }
    algo = UtilityExperiment(config)
    algo.run()
    return algo


def test_get_ith_weights_returns_first_series():
    algo = make_algo()
    w = algo.get_ith_weights(1)
    assert w is not None
    assert pytest.approx(w["score"]) == 1.0
    assert pytest.approx(w["quality"]) == -0.5


def test_get_normalized_solution_df_values():
    algo = make_algo()
    norm = algo.get_normalized_solution_df()
    assert pytest.approx(norm.loc[0, "score"]) == 1 / 3
    assert pytest.approx(norm.loc[0, "quality"]) == 1.0
    assert pytest.approx(norm.loc[1, "score"]) == 1.0
    assert pytest.approx(norm.loc[1, "quality"]) == 0.5


def test_get_util_uses_weights_and_normalization():
    algo = make_algo()
    weights = {"score": 1.0, "quality": -0.5}
    util0 = algo.get_util(0, weights)
    util1 = algo.get_util(1, weights)
    assert pytest.approx(util0, rel=1e-6) == -0.1666667
    assert pytest.approx(util1, rel=1e-6) == 0.75
