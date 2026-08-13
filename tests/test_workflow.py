"""Tests for the typed schema-v2 workflow."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.test_output_models import _all_outputs
from tradingscope.agents import workflow
from tradingscope.agents.output import (
    AnalysisResult,
    FundamentalsAnalystOutput,
    MarketAnalystOutput,
    NewsAnalystOutput,
    SocialMediaAnalystOutput,
    TraderOutput,
)
from tradingscope.agents.utils.structured_output import StructuredOutputValidationError
from tradingscope.utils import oss_structured_output_uploader as persistence


def _workflow_patches(runner: SimpleNamespace, research_orchestrator, risk_orchestrator):
    cache_usage = SimpleNamespace(log_summary=MagicMock())
    context = SimpleNamespace(
        non_thinking_model=object(),
        cache_usage=cache_usage,
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
    research_factory = stack.enter_context(
        patch.object(
            workflow,
            "create_research_debate_orchestrator",
            return_value=research_orchestrator,
        ),
    )
    risk_factory = stack.enter_context(
        patch.object(
            workflow,
            "create_debate_orchestrator",
            return_value=risk_orchestrator,
        ),
    )
    persist_node = stack.enter_context(patch.object(workflow, "persist_node_output", new=AsyncMock()))
    persist_result = stack.enter_context(patch.object(workflow, "persist_analysis_result", new=AsyncMock()))
    return stack, persist_node, persist_result, research_factory, risk_factory, cache_usage


@pytest.mark.asyncio
async def test_analyze_returns_fully_typed_result(tmp_path) -> None:
    market, fundamentals, news, social, research, trader, portfolio = _all_outputs()
    runner = SimpleNamespace(run=AsyncMock(side_effect=[market, fundamentals, news, social, trader]))
    research_orchestrator = SimpleNamespace(run_debate=AsyncMock(return_value=research))
    risk_orchestrator = SimpleNamespace(run_debate=AsyncMock(return_value=portfolio))

    stack, persist_node, persist_result, research_factory, risk_factory, cache_usage = _workflow_patches(
        runner,
        research_orchestrator,
        risk_orchestrator,
    )
    manifest_path = tmp_path / "data" / "2026-07-14" / "AAPL" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"status":"complete"}')
    with (
        stack,
        patch.object(persistence, "DEFAULT_CONFIG", {"results_dir": str(tmp_path)}),
        patch.object(persistence, "_get_client", return_value=None),
    ):
        result = await workflow.analyze("AAPL", "2026-07-14")

    assert isinstance(result, AnalysisResult)
    assert result.analysts.market is market
    assert result.research_manager is research
    assert result.trader is trader
    assert result.portfolio_manager is portfolio
    assert result.schema_version == "2.0"
    assert not manifest_path.exists()
    persisted_names = [call.args[0] for call in persist_node.await_args_list]
    assert all(call.kwargs == {"ticker": "AAPL", "trade_date": "2026-07-14"} for call in persist_node.await_args_list)
    assert set(persisted_names[:4]) == {
        "market_analyst",
        "fundamentals_analyst",
        "news_analyst",
        "social_media_analyst",
    }
    assert persisted_names[4:] == [
        "research_manager",
        "trader",
        "portfolio_manager",
    ]
    persist_result.assert_awaited_once_with(result)
    assert research_factory.call_args.kwargs["reference_outputs"] == (market, fundamentals, news, social)
    assert runner.run.await_args_list[4].kwargs["reference_outputs"] == (research, market, fundamentals, news, social)
    assert risk_factory.call_args.kwargs["reference_outputs"] == (
        trader,
        research,
        market,
        fundamentals,
        news,
        social,
    )
    cache_usage.log_summary.assert_called_once_with()


@pytest.mark.asyncio
async def test_analyst_failure_stops_downstream_work() -> None:
    failure = StructuredOutputValidationError("MarketAnalyst", [])
    runner = SimpleNamespace(run=AsyncMock(side_effect=failure))
    research_orchestrator = SimpleNamespace(run_debate=AsyncMock())
    risk_orchestrator = SimpleNamespace(run_debate=AsyncMock())

    stack, persist_node, persist_result, _, _, cache_usage = _workflow_patches(
        runner,
        research_orchestrator,
        risk_orchestrator,
    )
    with stack:
        with pytest.raises(StructuredOutputValidationError):
            await workflow.analyze("AAPL", "2026-07-14")

    research_orchestrator.run_debate.assert_not_awaited()
    risk_orchestrator.run_debate.assert_not_awaited()
    persist_node.assert_not_awaited()
    persist_result.assert_not_awaited()
    cache_usage.log_summary.assert_not_called()


@pytest.mark.asyncio
async def test_analyst_failure_cancels_and_collects_sibling_tasks() -> None:
    failure = StructuredOutputValidationError("MarketAnalyst", [])
    cancelled_agents: set[str] = set()

    async def run(agent, output_model):
        del output_model
        if agent.name == "create_market_analyst_agent":
            await asyncio.sleep(0)
            raise failure
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_agents.add(agent.name)
            raise

    runner = SimpleNamespace(run=run)
    research_orchestrator = SimpleNamespace(run_debate=AsyncMock())
    risk_orchestrator = SimpleNamespace(run_debate=AsyncMock())

    stack, _, _, _, _, _ = _workflow_patches(
        runner,
        research_orchestrator,
        risk_orchestrator,
    )
    with stack:
        with pytest.raises(StructuredOutputValidationError):
            await workflow.analyze("AAPL", "2026-07-14")

    assert cancelled_agents == {
        "create_fundamentals_analyst_agent",
        "create_news_analyst_agent",
        "create_social_media_analyst_agent",
    }


@pytest.mark.asyncio
async def test_each_analyst_is_persisted_as_soon_as_it_finishes() -> None:
    market, fundamentals, news, social, research, trader, portfolio = _all_outputs()
    outputs = {
        MarketAnalystOutput: market,
        FundamentalsAnalystOutput: fundamentals,
        NewsAnalystOutput: news,
        SocialMediaAnalystOutput: social,
        TraderOutput: trader,
    }
    release_slow_analysts = asyncio.Event()
    market_persisted = asyncio.Event()

    async def run(_agent, output_model, **_kwargs):
        if output_model is not MarketAnalystOutput and output_model is not TraderOutput:
            await release_slow_analysts.wait()
        return outputs[output_model]

    async def persist_node(name, _output, **_kwargs):
        if name == "market_analyst":
            market_persisted.set()

    runner = SimpleNamespace(run=AsyncMock(side_effect=run))
    research_orchestrator = SimpleNamespace(run_debate=AsyncMock(return_value=research))
    risk_orchestrator = SimpleNamespace(run_debate=AsyncMock(return_value=portfolio))
    stack, persist, _, _, _, _ = _workflow_patches(
        runner,
        research_orchestrator,
        risk_orchestrator,
    )
    persist.side_effect = persist_node

    with stack:
        analysis_task = asyncio.create_task(workflow.analyze("AAPL", "2026-07-14"))
        await asyncio.wait_for(market_persisted.wait(), timeout=1)
        assert not analysis_task.done()
        release_slow_analysts.set()
        await analysis_task
