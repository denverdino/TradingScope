"""Tests for typed AgentContext report state."""

from __future__ import annotations

from unittest.mock import patch

from agentscope.formatter import OpenAIResponseFormatter

from tests.test_output_models import _all_outputs
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.dashscope_response_model import DashScopeResponseModel
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


def test_context_uses_two_responses_models() -> None:
    with (
        patch("tradingscope.agents.utils.context.get_latest_us_trading_date", return_value="2026-08-10"),
        patch("tradingscope.agents.utils.context.OpenAICredential") as credential_factory,
        patch("tradingscope.agents.utils.context.DashScopeResponseModel") as response_model,
    ):
        response_model.Parameters = DashScopeResponseModel.Parameters
        AgentContext()

    assert response_model.call_count == 2
    assert response_model.call_args_list[0].kwargs["model"] == DEFAULT_CONFIG["deep_think_llm"]
    assert response_model.call_args_list[0].kwargs["parameters"].thinking_enable is True
    assert response_model.call_args_list[1].kwargs["parameters"].thinking_enable is False
    assert all(isinstance(call.kwargs["formatter"], OpenAIResponseFormatter) for call in response_model.call_args_list)
    assert response_model.call_args_list[0].kwargs["formatter"] is response_model.call_args_list[1].kwargs["formatter"]
    assert all(call.kwargs["client_kwargs"] == {"default_headers": {"x-dashscope-session-cache": "enable"}} for call in response_model.call_args_list)
    assert all(call.kwargs["stream"] is True for call in response_model.call_args_list)
    assert all(call.kwargs["credential"] is credential_factory.return_value for call in response_model.call_args_list)


def test_context_configures_responses_credential() -> None:
    with (
        patch("tradingscope.agents.utils.context.get_latest_us_trading_date", return_value="2026-08-10"),
        patch("tradingscope.agents.utils.context.OpenAICredential") as credential_factory,
        patch("tradingscope.agents.utils.context.DashScopeResponseModel"),
    ):
        AgentContext()

    assert credential_factory.call_args.kwargs["base_url"] == DEFAULT_CONFIG["backend_url"]
