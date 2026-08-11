"""Tests for typed AgentContext report state."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from agentscope.model import OpenAIResponseModel

from tests.test_output_models import _all_outputs
from tradingscope.agents.utils.context import AgentContext, CodeInterpreterModel
from tradingscope.default_config import DEFAULT_CONFIG


def _typed_context() -> AgentContext:
    market, fundamentals, news, social, research, trader, portfolio = _all_outputs()
    context = AgentContext.__new__(AgentContext)
    context.company_of_interest = "AAPL"
    context.trade_date = "2026-07-14"
    context.latest_trading_date = "2026-07-13"
    context.market_analysis = market
    context.fundamentals_analysis = fundamentals
    context.news_analysis = news
    context.social_analysis = social
    context.research_decision = research
    context.trader_decision = trader
    context.portfolio_decision = portfolio
    return context


def test_context_renders_analysts_from_typed_outputs() -> None:
    context = _typed_context()

    markdown = context.generate_analyst_reports_md()

    assert context.market_analysis.decision.summary in markdown
    assert context.fundamentals_analysis.decision.summary in markdown
    assert context.news_analysis.decision.summary in markdown
    assert context.social_analysis.decision.summary in markdown


def test_context_renders_typed_downstream_inputs() -> None:
    context = _typed_context()

    trader_context = context.generate_trader_context_md()
    risk_context = context.generate_risk_evaluation_context_md()

    assert context.research_decision.decision.summary in trader_context
    assert context.trader_decision.decision.summary in risk_context


def test_code_interpreter_uses_builtin_tools_model() -> None:
    with (
        patch("tradingscope.agents.utils.context.get_latest_us_trading_date", return_value="2026-08-10"),
        patch("tradingscope.agents.utils.context.DashScopeCredential"),
        patch("tradingscope.agents.utils.context.DashScopeChatModel"),
        patch("tradingscope.agents.utils.context.OpenAICredential"),
        patch("tradingscope.agents.utils.context.CodeInterpreterModel") as code_interpreter_model,
    ):
        AgentContext()

    assert code_interpreter_model.call_args.kwargs["model"] == DEFAULT_CONFIG["builtin_tools_model"]
    assert DEFAULT_CONFIG["builtin_tools_model"] == DEFAULT_CONFIG["deep_think_llm"]


def test_code_interpreter_uses_responses_api_builtin_tool() -> None:
    model = CodeInterpreterModel.__new__(CodeInterpreterModel)

    with patch.object(OpenAIResponseModel, "_call_api", new_callable=AsyncMock) as call_api:
        asyncio.run(model._call_api("qwen3.8-max", []))

    assert model._format_tools(None, None) == ([{"type": "code_interpreter"}], None)
    assert call_api.await_args.kwargs["extra_body"] == {"enable_thinking": True}
