"""Tests for the typed schema-v2 workflow."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.test_output_models import _all_outputs
from tradingscope.agents import workflow
from tradingscope.agents.output import AnalysisResult
from tradingscope.agents.utils.structured_output import StructuredOutputValidationError


def _workflow_patches(runner: SimpleNamespace, research_orchestrator, risk_orchestrator):
    context = SimpleNamespace(
        non_thinking_model=object(),
        company_of_interest="",
        trade_date="2026-07-14",
        latest_trading_date="2026-07-13",
    )
    stack = ExitStack()
    stack.enter_context(patch.object(workflow, "AgentContext", return_value=context))
    stack.enter_context(patch.object(workflow, "StructuredAgentRunner", return_value=runner, create=True))
    for name in (
        "create_market_analyst_agent",
        "create_fundamentals_analyst_agent",
        "create_news_analyst_agent",
        "create_social_media_analyst_agent",
        "create_bear_researcher_agent",
        "create_bull_researcher_agent",
        "create_research_manager_agent",
        "create_trader_agent",
        "create_aggressive_debator_agent",
        "create_conservative_debator_agent",
        "create_neutral_debator_agent",
        "create_portfolio_manager_agent",
    ):
        stack.enter_context(patch.object(workflow, name, return_value=SimpleNamespace(name=name)))
    stack.enter_context(
        patch.object(
            workflow,
            "create_research_debate_orchestrator",
            return_value=research_orchestrator,
        ),
    )
    stack.enter_context(
        patch.object(
            workflow,
            "create_debate_orchestrator",
            return_value=risk_orchestrator,
        ),
    )
    persist = stack.enter_context(patch.object(workflow, "persist_analysis_result", new=AsyncMock()))
    return stack, persist


@pytest.mark.asyncio
async def test_analyze_returns_fully_typed_result() -> None:
    market, fundamentals, news, social, research, trader, portfolio = _all_outputs()
    runner = SimpleNamespace(run=AsyncMock(side_effect=[market, fundamentals, news, social, trader]))
    research_orchestrator = SimpleNamespace(run_debate=AsyncMock(return_value=research))
    risk_orchestrator = SimpleNamespace(run_debate=AsyncMock(return_value=portfolio))

    stack, persist = _workflow_patches(runner, research_orchestrator, risk_orchestrator)
    with stack:
        result = await workflow.analyze("AAPL", "2026-07-14")

    assert isinstance(result, AnalysisResult)
    assert result.analysts.market is market
    assert result.research_manager is research
    assert result.trader is trader
    assert result.portfolio_manager is portfolio
    assert result.schema_version == "2.0"
    persist.assert_awaited_once_with(result)


@pytest.mark.asyncio
async def test_analyst_failure_stops_downstream_work() -> None:
    failure = StructuredOutputValidationError("MarketAnalyst", [])
    runner = SimpleNamespace(run=AsyncMock(side_effect=failure))
    research_orchestrator = SimpleNamespace(run_debate=AsyncMock())
    risk_orchestrator = SimpleNamespace(run_debate=AsyncMock())

    stack, persist = _workflow_patches(runner, research_orchestrator, risk_orchestrator)
    with stack:
        with pytest.raises(StructuredOutputValidationError):
            await workflow.analyze("AAPL", "2026-07-14")

    research_orchestrator.run_debate.assert_not_awaited()
    risk_orchestrator.run_debate.assert_not_awaited()
    persist.assert_not_awaited()
