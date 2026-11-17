"""Tests for the scheduled trading analysis workflow."""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW_PATH = Path(__file__).parents[1] / ".github" / "workflows" / "trading-analysis.yml"
TICKERS = ["MSFT", "AMZN", "INTC", "NVDA", "GOOGL", "BABA", "TSM"]


def _run_command(job: dict[str, object], step_name: str) -> str:
    steps = job["steps"]
    assert isinstance(steps, list)
    step = next(step for step in steps if step["name"] == step_name)
    return step["run"]


def test_workflow_runs_at_most_two_tickers_after_single_evaluation() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())

    evaluate = workflow["jobs"]["evaluate"]
    analyze = workflow["jobs"]["analyze"]

    assert analyze["needs"] == "evaluate"
    assert analyze["strategy"] == {
        "fail-fast": False,
        "max-parallel": 2,
        "matrix": {"ticker": TICKERS},
    }
    evaluate_command = _run_command(evaluate, "Evaluate stock analysis")
    assert "tradingscope.evaluate" in evaluate_command
    assert f"tickers=({' '.join(TICKERS)})" in evaluate_command
    assert 'tradingscope.main "${{ matrix.ticker }}"' in _run_command(analyze, "Run stock analysis")
