from pathlib import Path
import sys

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment import Experiment


def test_to_json_serializes_state():
    df = pd.DataFrame(
        [
            {"score": 1.0, "quality": 2.0},
            {"score": 3.5, "quality": 0.5},
        ]
    )
    config = {
        "solutions_df": df,
        "metric_columns": ["score", "quality"],
        "iterations": 4,
        "seed": 42,
        "llm_params": {"temperature": 0.1},
        "gtu_weights": {"score": 1.5, "quality": -0.25},
        "human_sol": [1, 0],
    }
    algo = Experiment(config)
    algo._history = {"step": "done"}
    algo._plot_data = {"plot": [1, 2]}
    algo._winner_idx = 1
    algo._winner_solution = df.iloc[1].to_dict()
    algo._nar = 3
    algo.prompt_history = [{"prompt": "p", "response": "r"}]

    payload = algo.to_json()

    assert payload["algo"] == "Experiment"
    assert payload["config"]["iterations"] == 4
    assert payload["config"]["metric_columns"] == ["score", "quality"]
    assert payload["config"]["seed"] == 42
    assert payload["config"]["llm_params"] == {"temperature": 0.1}
    assert payload["config"]["gtu_weights"] == {"score": 1.5, "quality": -0.25}
    assert payload["config"]["human_sol"] == [1, 0]
    assert payload["solutions"] == df.to_dict(orient="records")
    assert payload["history"] == {"step": "done"}
    assert payload["plot_data"] == {"plot": [1, 2]}
    assert payload["winner_idx"] == 1
    assert payload["winner_solution"] == df.iloc[1].to_dict()
    assert payload["nar"] == 3
    assert payload["prompt_history"] == [{"prompt": "p", "response": "r"}]


def test_from_json_restores_config_and_state():
    solutions = [
        {"metric": 5.0, "other": 1.0},
        {"metric": 2.0, "other": 3.0},
    ]
    data = {
        "config": {
            "metric_columns": ["metric"],
            "iterations": 3,
            "gtu_weights": {"metric": 1.2},
        },
        "solutions": solutions,
        "results": {
            "history": {"step": "restored"},
            "plot_data": {"points": [1]},
            "final_winner_idx": 1,
            "final_winner_solution": solutions[1],
            "nar": 2,
            "prompt_history": [{"prompt": "p"}],
        },
    }

    algo = Experiment.from_json(data)

    assert_frame_equal(algo.solutions_df, pd.DataFrame(solutions))
    assert algo.metric_columns == ["metric"]
    assert algo.iterations == 3
    assert algo.gtu_weights == {"metric": 1.2}
    assert algo.human_sol == [0, 1]
    assert algo._history == {"step": "restored"}
    assert algo._plot_data == {"points": [1]}
    assert algo._winner_idx == 1
    assert algo._winner_solution == solutions[1]
    assert algo._nar == 2
    assert algo.prompt_history == [{"prompt": "p"}]


def test_get_gtu_computes_weighted_sum():
    df = pd.DataFrame(
        [
            {"m1": 1.0, "m2": 5.0},
            {"m1": 4.0, "m2": 1.0},
        ]
    )
    algo = Experiment({
        "solutions_df": df,
        "metric_columns": ["m1", "m2"],
        "gtu_weights": {"m1": 2, "m2": -1},
    })

    assert algo.get_gtu(1) == pytest.approx(7.0)
    assert algo.get_gtu(0) == pytest.approx(-3.0)


def test_get_nar_uses_rank_and_dataset_size():
    df = pd.DataFrame(
        [
            {"score": 1},
            {"score": 2},
            {"score": 3},
        ]
    )
    algo = Experiment({"solutions_df": df, "human_sol": [1, 0, 2]})
    algo._winner_idx = 0

    assert algo.get_nar() == pytest.approx(2 / 3)


def test_get_gtu_raises_on_invalid_data():
    df = pd.DataFrame([{"m1": 1}])
    algo = Experiment({"solutions_df": df, "metric_columns": ["m1"], "gtu_weights": {"m1": 1}})
    with pytest.raises(IndexError):
        algo.get_gtu(999)  # out of bounds


def test_get_nar_fallback_rank_when_winner_not_in_human_sol():
    df = pd.DataFrame([{"s": 1}, {"s": 2}, {"s": 3}, {"s": 4}])
    algo = Experiment({"solutions_df": df, "human_sol": [3, 1]})
    algo._winner_idx = 2  # not in human_sol

    # _lookup_rank falls back to (m + 1 + N) / 2 = (2 + 1 + 4) / 2 = 3.5
    # nar = 3.5 / 4 = 0.875
    assert algo.get_nar() == pytest.approx(0.875)


def test_get_nar_raises_without_dataframe():
    algo_no_df = Experiment({})
    with pytest.raises((TypeError, AttributeError)):
        algo_no_df.get_nar()
