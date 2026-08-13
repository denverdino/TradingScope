"""Contract tests for the AgentScope structured-output API."""

from __future__ import annotations

import inspect
from importlib.metadata import version
from types import SimpleNamespace
from unittest.mock import MagicMock

from packaging.version import Version

from tradingscope.agents.utils.dashscope_response_model import DashScopeResponseModel


def test_agentscope_exposes_structured_output_api() -> None:
    assert Version(version("agentscope")) >= Version("2.0.6")

    method = DashScopeResponseModel.generate_structured_output
    parameters = inspect.signature(method).parameters

    assert inspect.iscoroutinefunction(method)
    assert "messages" in parameters
    assert "structured_model" in parameters


def test_agent_context_creates_cache_and_tracing_middlewares(monkeypatch) -> None:
    from tradingscope.agents.utils import context as context_module
    from tradingscope.agents.utils.cache_usage import CacheUsageMiddleware

    sentinel_middlewares = [object()]
    monkeypatch.setattr(
        context_module,
        "create_tracing_middlewares",
        lambda: sentinel_middlewares,
    )
    monkeypatch.setattr(context_module, "get_latest_us_trading_date", lambda: "2026-07-21")
    monkeypatch.setattr(context_module, "OpenAICredential", lambda **_kwargs: object())
    response_model = MagicMock()
    response_model.Parameters = context_module.DashScopeResponseModel.Parameters
    monkeypatch.setattr(context_module, "DashScopeResponseModel", response_model)

    context = context_module.AgentContext()

    assert isinstance(context.middlewares[0], CacheUsageMiddleware)
    assert context.middlewares[0].collector is context.cache_usage
    assert context.middlewares[1:] == sentinel_middlewares


def test_tool_using_factory_receives_context_middlewares(monkeypatch) -> None:
    from tradingscope.agents.analysts import social_media_analyst

    captured = {}
    context = SimpleNamespace(
        company_of_interest="AAPL",
        trade_date="2026-07-21",
        latest_trading_date="2026-07-20",
        model=object(),
        middlewares=[object()],
    )
    market_info = {
        "market_name": "US",
        "currency_name": "US dollar",
        "currency_symbol": "$",
    }
    monkeypatch.setattr(social_media_analyst.StockUtils, "get_market_info", lambda _ticker: market_info)
    monkeypatch.setattr(social_media_analyst, "get_company_name", lambda _ticker, _info: "Apple")
    monkeypatch.setattr(
        social_media_analyst,
        "Agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    social_media_analyst.create_social_media_analyst_agent(context)

    assert captured["middlewares"] is context.middlewares


def test_plain_factory_receives_context_middlewares(monkeypatch) -> None:
    from tradingscope.agents.managers import portfolio_manager

    captured = {}
    context = SimpleNamespace(
        company_of_interest="AAPL",
        trade_date="2026-07-21",
        latest_trading_date="2026-07-20",
        model=object(),
        middlewares=[object()],
        generate_risk_evaluation_context_md=lambda: "risk context",
    )
    monkeypatch.setattr(
        portfolio_manager,
        "Agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    portfolio_manager.create_portfolio_manager_agent(context)

    assert captured["middlewares"] is context.middlewares
