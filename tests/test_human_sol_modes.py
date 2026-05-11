import sys
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment import Experiment
from tournament_algorithm import TournamentExperiment
from run_algorithm import load_config, resolve_mode


def _build_algo_for_mode(scenario: str, mode: str):
    cfg = load_config(scenario)
    resolve_mode(cfg, mode)
    mode_def = (cfg.get("modes") or {}).get(mode) or {}
    data_path = Path(__file__).resolve().parent.parent / cfg["data_csv"]
    df = pd.read_csv(data_path).dropna(subset=cfg["metric_columns"])
    human_sol = mode_def.get("human_sol") or cfg.get("human_sol") or []
    algo = Experiment(
        {"solutions_df": df, "metric_columns": cfg["metric_columns"], "human_sol": human_sol}
    )
    return algo, human_sol


OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def _find_output_file(scenario: str, mode: Optional[str] = None, algo_type: str = "tournament") -> Optional[Path]:
    """Find an output file matching the given scenario/mode/algo in any flat output directory."""
    for output_dir in OUTPUT_ROOT.iterdir():
        if not output_dir.is_dir():
            continue
        for json_file in output_dir.glob("*.json"):
            try:
                with json_file.open() as f:
                    data = json.load(f)
                meta = data.get("meta") or {}
                file_scenario = meta.get("scenario") or meta.get("tag")
                file_mode = meta.get("mode")
                file_algo = meta.get("algo", "").lower()

                if file_scenario != scenario:
                    continue
                if mode and file_mode != mode:
                    continue
                if algo_type and algo_type.lower() not in file_algo:
                    continue
                return json_file
            except Exception:
                continue
    return None


def _load_algo_from_output(output_path: Path) -> TournamentExperiment:
    return Experiment.from_json(output_path)


@pytest.mark.parametrize(
    ("scenario", "mode", "expected_prefix"),
    [
        ("flight00", "Complicated_structured", [593, 685, 602, 612, 692, 603, 617, 481, 1, 276, 138, 5, 426, 285, 271, 7]),
        ("flight02", "Complicated", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ],
)
def test_get_rank_respects_human_sol_order_for_curated_modes(scenario, mode, expected_prefix):
    algo, human_sol = _build_algo_for_mode(scenario, mode)

    assert human_sol, "Expected curated human_sol list from mode definition"
    assert algo.human_sol == human_sol

    for position, idx in enumerate(expected_prefix, start=1):
        assert algo.get_rank(idx) == position

    assert algo.get_rank(human_sol[-1]) == len(human_sol)


@pytest.mark.parametrize(
    ("scenario", "mode"),
    [
        ("exam", "REGISTRAR"),
    ],
)
def test_get_winner_returns_final_winner_from_saved_results(scenario, mode):
    output_path = _find_output_file(scenario, mode, "tournament")
    if output_path is None:
        pytest.skip(f"No output file found for {scenario}/{mode}")

    algo = _load_algo_from_output(output_path)

    # Verify we can use get_algo, get_scenario, get_mode
    assert algo.get_algo() == "TournamentExperiment"
    assert algo.get_scenario() == scenario
    assert algo.get_mode() == mode

    winner = algo.get_winner()
    assert winner is not None


@pytest.mark.parametrize(
    ("scenario", "mode", "batch_num"),
    [
        ("exam", "REGISTRAR", 1),
    ],
)
def test_get_ith_winner_returns_batch_winner(scenario, mode, batch_num):
    output_path = _find_output_file(scenario, mode, "tournament")
    if output_path is None:
        pytest.skip(f"No output file found for {scenario}/{mode}")

    algo = _load_algo_from_output(output_path)

    winner = algo.get_ith_winner(batch_num)
    assert winner is not None


def test_get_rank_uses_registrar_human_sol():
    algo, human_sol = _build_algo_for_mode("exam", "REGISTRAR")

    expected = [1519, 4344, 2091, 2620, 1527, 3444, 71, 4719, 2194, 4771, 4392]
    assert human_sol == expected
    assert algo.human_sol == expected

    for pos, idx in enumerate(expected, start=1):
        assert algo.get_rank(idx) == pos

    assert algo.get_rank(expected[-1]) == len(expected)


def test_get_rank_defaults_when_human_sol_missing():
    # BASE mode has no human_sol; verify rank fallback to (1 + N) / 2.
    algo, human_sol = _build_algo_for_mode("exam", "BASE")

    assert human_sol == []
    assert algo.human_sol == []

    # When human_sol is empty (m=0), fallback is (0 + 1 + N) / 2
    N = len(algo.solutions_df)
    expected_fallback = (1 + N) / 2
    assert algo.get_rank(10) == expected_fallback
    assert algo.get_rank(N - 1) == expected_fallback


@pytest.mark.parametrize(
    ("mode", "expected_prefix"),
    [
        ("STUDENT", [23, 2, 56, 29, 66]),
        ("STUDENT_HARD", [4, 22, 1, 2, 74]),
    ],
)
def test_get_rank_respects_headphones_human_sol(mode, expected_prefix):
    algo, human_sol = _build_algo_for_mode("headphones", mode)

    assert human_sol, "Expected curated human_sol for headphones mode"
    assert algo.human_sol == human_sol

    for position, idx in enumerate(expected_prefix, start=1):
        assert algo.get_rank(idx) == position

    assert algo.get_rank(human_sol[-1]) == len(human_sol)
