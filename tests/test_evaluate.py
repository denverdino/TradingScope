"""Tests for the post-market evaluation CLI."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


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
