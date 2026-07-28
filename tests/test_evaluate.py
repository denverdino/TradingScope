"""Tests for the post-market evaluation CLI."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


def _progressive_results() -> list:
    from tradingscope.agents.evaluation.models import EvaluationResult

    return [
        EvaluationResult(
            ticker="AAPL",
            evaluation="one-day evaluation",
            lesson="one-day lesson",
            trade_date="2025-01-20",
            horizon_days=1,
            status="correct",
        ),
        EvaluationResult(
            ticker="AAPL",
            evaluation="three-day evaluation",
            lesson="three-day lesson",
            trade_date="2025-01-20",
            horizon_days=3,
            status="inconclusive",
        ),
        EvaluationResult(
            ticker="AAPL",
            evaluation="next-day evaluation",
            lesson="next-day lesson",
            trade_date="2025-01-21",
            horizon_days=1,
            status="inconclusive",
        ),
    ]


def test_main_owns_tracing_lifecycle() -> None:
    from tradingscope import evaluate as evaluate_module

    provider = object()
    setup_tracing = Mock(return_value=provider)
    shutdown_tracing = Mock()
    process = Mock()
    with (
        patch.object(evaluate_module, "setup_tracing", setup_tracing),
        patch.object(evaluate_module, "shutdown_tracing", shutdown_tracing),
        patch.object(evaluate_module, "_main", process),
    ):
        evaluate_module.main()

    process.assert_called_once_with()
    setup_tracing.assert_called_once_with("tradingscope-evaluation")
    shutdown_tracing.assert_called_once_with(provider)


def test_run_passes_context_middlewares_to_evaluator() -> None:
    from tradingscope import evaluate as evaluate_module

    context = SimpleNamespace(non_thinking_model=object(), middlewares=[object()])
    evaluator = Mock()
    evaluator.run_batch_evaluation = AsyncMock(return_value=[])
    evaluator_class = Mock(return_value=evaluator)

    with (
        patch(
            "tradingscope.agents.utils.context.AgentContext",
            return_value=context,
        ),
        patch(
            "tradingscope.agents.evaluation.evaluator.AnalysisEvaluator",
            evaluator_class,
        ),
    ):
        asyncio.run(
            evaluate_module._run(
                tickers=["AAPL"],
                date=None,
                results_dir=None,
            ),
        )

    evaluator_class.assert_called_once_with(
        model=context.non_thinking_model,
        memory_manager=None,
        results_dir=None,
        dry_run=False,
        middlewares=context.middlewares,
    )
    assert evaluator_class.call_args.kwargs["middlewares"] is context.middlewares


def test_run_labels_progressive_results_with_trade_date_horizon_and_status() -> None:
    from tradingscope import evaluate as evaluate_module

    context = SimpleNamespace(non_thinking_model=object(), middlewares=[])
    evaluator = Mock()
    evaluator.run_batch_evaluation = AsyncMock(return_value=_progressive_results())
    logger = Mock()

    with (
        patch(
            "tradingscope.agents.utils.context.AgentContext",
            return_value=context,
        ),
        patch(
            "tradingscope.agents.evaluation.evaluator.AnalysisEvaluator",
            return_value=evaluator,
        ),
        patch.object(evaluate_module, "logger", logger),
    ):
        asyncio.run(
            evaluate_module._run(
                tickers=["AAPL"],
                date=None,
                results_dir=None,
            ),
        )

    rendered_logs = "\n".join(str(call) for call in logger.info.call_args_list)
    assert "[AAPL|2025-01-20|1日|correct]" in rendered_logs
    assert "[AAPL|2025-01-20|3日|inconclusive]" in rendered_logs
    assert "[AAPL|2025-01-21|1日|inconclusive]" in rendered_logs


def test_evaluation_email_groups_progressive_results_by_trade_date() -> None:
    from tradingscope import evaluate as evaluate_module

    send_html_email = Mock()
    with (
        patch.dict(
            "os.environ",
            {"EMAIL_FROM": "sender@example.com", "EMAIL_PASSWORD": "secret"},
        ),
        patch(
            "tradingscope.utils.email_utils.send_html_email",
            send_html_email,
        ),
    ):
        evaluate_module._send_evaluation_email(
            _progressive_results(),
            "2025-01-21",
            ["recipient@example.com"],
        )

    html_content = send_html_email.call_args.args[1]
    assert html_content.count("<h3>分析日期：2025-01-20</h3>") == 1
    assert html_content.count("<h3>分析日期：2025-01-21</h3>") == 1
    assert "<h4>[1日|correct]</h4>" in html_content
    assert "<h4>[3日|inconclusive]</h4>" in html_content
