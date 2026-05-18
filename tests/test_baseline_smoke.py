"""End-to-end smoke test for the baseline algorithm via the CLI.

The baseline algorithm has no LLM dependency, so this test exercises the
full run_algorithm.py pipeline (config load → experiment instantiation →
iteration loop → JSON output) without needing API keys or SDK installs.

This is the only test in the suite that subprocesses run_algorithm.py; the
rest construct algorithm objects directly. Catches integration regressions
that unit tests miss (CLI parsing, output filename, payload structure).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ALGO = REPO_ROOT / "run_algorithm.py"


def test_baseline_cli_end_to_end(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        str(RUN_ALGO),
        "--algo", "baseline",
        "--scenario", "flights_ithaca_reston",
        "--mode", "Complicated",
        "--api-model", "groq",  # baseline never instantiates the client
        "--iterations", "3",
        "--seed", "42",
        "--output-root", str(tmp_path),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"run_algorithm.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    outputs = list(tmp_path.glob("*.json"))
    assert len(outputs) == 1, f"expected one JSON, got {outputs}"

    payload = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert "meta" in payload and "results" in payload

    meta = payload["meta"]
    assert meta["algo"] == "baseline"
    assert meta["scenario"] == "flights_ithaca_reston"
    assert meta["mode"] == "Complicated"
    assert meta["max_iters"] == 3
    assert meta["seed"] == 42
    assert meta["nar"] is not None, "NAR should be populated"

    # final_winner_idx must be a valid row in the flights_ithaca_reston data
    winner = payload["results"].get("final_winner_idx")
    assert isinstance(winner, int) and winner >= 0
