"""Tests for the scheduled trading analysis workflow."""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW_PATH = Path(__file__).parents[1] / ".github" / "workflows" / "trading-analysis.yml"
CI_PATH = WORKFLOW_PATH.with_name("ci.yml")
TICKERS = ["MSFT", "AMZN", "INTC", "NVDA", "GOOGL", "BABA", "TSM"]


def _run_command(job: dict[str, object], step_name: str) -> str:
    steps = job["steps"]
    assert isinstance(steps, list)
    step = next(step for step in steps if step["name"] == step_name)
    return step["run"]


def test_ci_runs_only_offline_quality_checks_without_secrets() -> None:
    text = CI_PATH.read_text()
    workflow = yaml.safe_load(text)

    # PyYAML's YAML 1.1 loader interprets GitHub's unquoted `on` as True.
    assert workflow.get("on", workflow.get(True)) == {"push": None, "pull_request": None}
    assert workflow["permissions"] == {"contents": "read"}
    assert "secrets." not in text
    assert "env" not in workflow
    assert len(workflow["jobs"]) == 1

    job = next(iter(workflow["jobs"].values()))
    assert job["runs-on"] == "ubuntu-latest"
    assert "permissions" not in job
    assert "env" not in job
    steps = job["steps"]
    assert [step["uses"] for step in steps if "uses" in step] == [
        "actions/checkout@v6",
        "actions/setup-python@v6",
        "astral-sh/setup-uv@v7",
    ]
    python_setup = next(step for step in steps if step.get("uses") == "actions/setup-python@v6")
    assert python_setup["with"]["python-version"] == "3.13"
    assert [step["run"].strip() for step in steps if "run" in step] == [
        "uv sync --locked --extra dev",
        "make lint",
        "make format-check",
        "RUN_DASHSCOPE_INTEGRATION=0 uv run pytest -q",
    ]
    assert all("continue-on-error" not in step and "env" not in step for step in steps)
    assert "continue-on-error" not in job


def test_scheduled_workflow_reports_failures_and_analyzes_unless_cancelled() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    evaluate = workflow["jobs"]["evaluate"]
    analyze = workflow["jobs"]["analyze"]

    evaluate_command = _run_command(evaluate, "Evaluate stock analysis")
    analyze_command = _run_command(analyze, "Run stock analysis")
    assert "||" not in evaluate_command
    assert "|| true" not in analyze_command
    assert evaluate_command.splitlines()[-1] == ('uv run python -m tradingscope.evaluate --tickers="$ticker_list" --email-to="${{ secrets.EMAIL_TO }}"')
    assert analyze_command.strip() == (
        "uv run python -m tradingscope.main \"${{ matrix.ticker }}\" --email-to=\"${{ secrets.EMAIL_TO }}\" ${{ inputs.force && '--force' || '' }}"
    )
    assert analyze["needs"] == "evaluate"
    assert analyze["if"] == "${{ !cancelled() }}"
    for job in (evaluate, analyze):
        assert "continue-on-error" not in job
        assert all("continue-on-error" not in step for step in job["steps"])


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
    assert 'ticker_list=$(IFS=,; echo "${tickers[*]}")' in evaluate_command
    assert 'tradingscope.main "${{ matrix.ticker }}"' in _run_command(analyze, "Run stock analysis")
    assert workflow.get("on", workflow.get(True)) == {
        "schedule": [{"cron": "0 0 * * *"}],
        "workflow_dispatch": {
            "inputs": {
                "force": {
                    "description": "Analyze even when no newer completed session exists",
                    "type": "boolean",
                    "default": False,
                },
            },
        },
    }
    expected_env = {
        "DASHSCOPE_API_KEY": "${{ secrets.DASHSCOPE_API_KEY }}",
        "DASHSCOPE_BASE_URL": "${{ secrets.DASHSCOPE_BASE_URL }}",
        "ALPHA_VANTAGE_API_KEY": "${{ secrets.ALPHA_VANTAGE_API_KEY }}",
        "EMAIL_FROM": "${{ secrets.EMAIL_FROM }}",
        "EMAIL_PASSWORD": "${{ secrets.EMAIL_PASSWORD }}",
        "PERPLEXITY_API_KEY": "${{ secrets.PERPLEXITY_API_KEY }}",
        "OSS_ACCESS_KEY_ID": "${{ secrets.OSS_ACCESS_KEY_ID }}",
        "OSS_ACCESS_KEY_SECRET": "${{ secrets.OSS_ACCESS_KEY_SECRET }}",
        "OSS_REGION": "${{ secrets.OSS_REGION }}",
        "OSS_BUCKET": "${{ secrets.OSS_BUCKET }}",
        "TZ": "Asia/Shanghai",
    }
    for job, step_name in ((evaluate, "Evaluate stock analysis"), (analyze, "Run stock analysis")):
        assert job["runs-on"] == "ubuntu-latest"
        step = next(step for step in job["steps"] if step["name"] == step_name)
        assert step["env"] == expected_env
        assert _run_command(job, "Install dependencies").strip() == "uv sync --extra dev"
