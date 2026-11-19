"""Tests for the deterministic evaluation CLI."""

from unittest.mock import Mock, patch

import pytest

from tests.support.evaluation import evaluation_record
from tradingscope import evaluate as evaluate_module
from tradingscope.agents.evaluation.track_record import build_track_record


def _records():
    return [evaluation_record()]


def test_main_owns_tracing_lifecycle():
    provider = object()
    with (
        patch.object(evaluate_module, "setup_tracing", return_value=provider) as setup,
        patch.object(evaluate_module, "shutdown_tracing") as shutdown,
        patch.object(evaluate_module, "_main") as process,
    ):
        evaluate_module.main()
    setup.assert_called_once_with("tradingscope-evaluation")
    shutdown.assert_called_once_with(provider)
    process.assert_called_once_with()


def test_cli_no_model_context_is_created():
    store = Mock()
    evaluator = Mock()
    evaluator.run.return_value = []
    with (
        patch("tradingscope.agents.evaluation.oss_store.default_evaluation_store", return_value=store),
        patch("tradingscope.agents.evaluation.evaluator.AnalysisEvaluator", return_value=evaluator) as factory,
        patch("tradingscope.agents.utils.context.AgentContext") as context,
    ):
        evaluate_module._run(["AAPL"], None, None)
    context.assert_not_called()
    factory.assert_called_once_with(store, results_dir=None, dry_run=False)
    evaluator.run.assert_called_once_with(["AAPL"], analysis_date=None)


def test_no_mature_results_are_reported():
    evaluator = Mock()
    evaluator.run.return_value = []
    with (
        patch("tradingscope.agents.evaluation.oss_store.default_evaluation_store"),
        patch("tradingscope.agents.evaluation.evaluator.AnalysisEvaluator", return_value=evaluator),
        patch.object(evaluate_module, "logger") as logger,
    ):
        evaluate_module._run(["AAPL"], None, None)
    logger.info.assert_any_call("No mature evaluation results available")


def test_dry_run_with_email_requested_never_sends_email():
    evaluator = Mock()
    evaluator.run.return_value = _records()
    store = Mock()
    store.list_records.return_value = []
    with (
        patch("tradingscope.agents.evaluation.oss_store.default_evaluation_store", return_value=store),
        patch("tradingscope.agents.evaluation.evaluator.AnalysisEvaluator", return_value=evaluator),
        patch.object(evaluate_module, "_send_evaluation_email") as email,
    ):
        evaluate_module._run(["AAPL"], None, None, dry_run=True, email_to=["recipient@example.com"])
    email.assert_not_called()


def test_email_contains_only_deterministic_results():
    records = _records()
    track = build_track_record(records, min_samples=10, confidence_buckets=(0.5, 0.7))
    with (
        patch.dict("os.environ", {"EMAIL_FROM": "sender@example.com", "EMAIL_PASSWORD": "test-secret"}),
        patch("tradingscope.utils.email_utils.send_html_email") as send,
    ):
        evaluate_module._send_evaluation_email(records, track, "2026-07-25", ["recipient@example.com"])
    html = send.call_args.args[1]
    assert html.count("<h3>分析日期：2026-07-14</h3>") == 1
    assert "[1交易日|clean]" in html
    assert "市场分析师" in html
    assert "投资组合经理" in html
    assert "Track Record" in html
    assert "教训" not in html


def test_backfill_option_is_no_longer_supported():
    with patch("sys.argv", ["evaluate", "--tickers", "AAPL", "--date", "2026-07-14", "--backfill"]), pytest.raises(SystemExit):
        evaluate_module._main()
