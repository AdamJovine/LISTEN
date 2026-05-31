import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment import Experiment
from tournament_algorithm import TournamentExperiment
from run_algorithm import load_config


def _excel_label_to_index(label: str) -> int:
    """Convert Excel-style column label to 0-based index (A->0, B->1, ..., Z->25, AA->26, etc.)."""
    label = label.upper()
    result = 0
    for char in label:
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result - 1


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def _find_output_files(scenario: str, mode: Optional[str] = None, algo_type: str = "tournament") -> List[Path]:
    """Find all output files matching the given scenario/mode/algo in any flat output directory."""
    matches = []
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
                matches.append(json_file)
            except Exception:
                continue
    return sorted(matches)


def _load_results(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    return data.get("results") or data


def _load_df_for_scenario(scenario: str) -> tuple[pd.DataFrame, list[str]]:
    cfg = load_config(scenario)
    metric_cols = cfg["metric_columns"]
    df = pd.read_csv(PROJECT_ROOT / cfg["data_csv"]).dropna(subset=metric_cols).reset_index(drop=True)
    return df, metric_cols


def _values_close(a: Any, b: Any) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    # exact match for non-numerics
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    return a == pytest.approx(b, rel=1e-9, abs=1e-9)


def test_batch_metrics_and_letter_match_dataset():
    """Guard against regressions in JSON => df alignment and letter parsing."""
    # Find any exam file with REGISTRAR mode
    files = _find_output_files("exam", "REGISTRAR", "tournament")
    if not files:
        pytest.skip("No exam/REGISTRAR output files found")

    results = _load_results(files[0])
    history = results.get("history")
    if not history or "batch_comparisons" not in history:
        pytest.skip("No batch_comparisons in output file")

    batch_comparisons = history["batch_comparisons"]
    if not batch_comparisons:
        pytest.skip("Empty batch_comparisons")

    # Test letter mapping on first batch
    batch = batch_comparisons[0]
    letter = batch.get("choice_letter")
    if letter:
        letter_idx = _excel_label_to_index(letter)
        batch_indices = batch.get("batch_indices", [])
        if batch_indices:
            assert 0 <= letter_idx < len(batch_indices)
            assert batch["winner_idx"] == batch_indices[letter_idx]

    # Metrics match the original dataset row
    df, _ = _load_df_for_scenario("exam")
    winner_idx = batch.get("winner_idx")
    if winner_idx is not None and "winner_solution" in batch:
        row = df.iloc[winner_idx].to_dict()
        for key, val in batch["winner_solution"].items():
            expected = row.get(key)
            if expected is not None:
                assert _values_close(val, expected), f"Mismatch for {key}: {val} vs {expected}"


def test_excel_letter_selection_and_metrics_across_scenarios():
    """Run across available scenarios to stress letter parsing and winner solutions."""
    scenarios_to_check = ["exam"]  # Add more as they become available
    total_checked = 0
    index_keys = {"Unnamed: 0", "Unnamed: 0.1"}

    all_files = []
    for scenario in scenarios_to_check:
        files = _find_output_files(scenario, algo_type="tournament")
        if not files:
            continue
        all_files.extend([(scenario, f) for f in files])

    if not all_files:
        pytest.skip("No tournament output files available for configured scenarios")

    for scenario, path in all_files:
        try:
            df, metric_columns = _load_df_for_scenario(scenario)
        except Exception:
            continue

        results = _load_results(path)
        history = results.get("history")
        if not history or "batch_comparisons" not in history:
            continue
        meta_metrics = history.get("metadata", {}).get("metric_columns") or metric_columns

        for comp in history["batch_comparisons"]:
            letter = comp.get("choice_letter")
            batch_indices = comp.get("batch_indices") or []
            if not letter or not batch_indices:
                continue

            idx_in_batch = _excel_label_to_index(letter)
            assert idx_in_batch is not None
            assert 0 <= idx_in_batch < len(batch_indices)

            winner_idx = batch_indices[idx_in_batch]
            assert comp["winner_idx"] == winner_idx

            row = df.iloc[winner_idx].to_dict()
            shared_keys = (set(row.keys()) & set(comp["winner_solution"].keys())) - index_keys
            for key in shared_keys:
                val = comp["winner_solution"][key]
                expected = row[key]
                assert _values_close(val, expected), f"{scenario}::{path.name} mismatch for {key}"

            for key in meta_metrics:
                if key in comp["winner_solution"] and key in row:
                    assert _values_close(comp["winner_solution"][key], row[key])

            total_checked += 1

    # Require at least some comparisons to be checked
    assert total_checked >= 1, f"Only checked {total_checked} comparisons; expected at least 1"


def test_get_algo_returns_correct_class_name():
    """Test that get_algo returns the correct algorithm class name."""
    files = _find_output_files("exam", algo_type="tournament")
    if not files:
        pytest.skip("No exam output files found")

    with files[0].open() as f:
        data = json.load(f)

    algo = Experiment.from_json(data)
    assert algo.get_algo() == "TournamentExperiment"
    assert algo.get_scenario() == "exam"
    assert algo.get_mode() is not None
