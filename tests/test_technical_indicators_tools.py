import inspect
import os
from unittest.mock import patch

import pytest

from tradingscope.agents.utils import technical_indicators_tools
from tradingscope.dataflows import interface

EXPECTED_INDICATORS = (
    "close_10_ema",
    "rsi",
    "macd",
    "macds",
    "macdh",
    "atr",
    "boll",
    "boll_ub",
    "boll_lb",
)


def test_get_indicators_returns_fixed_batch_in_order(monkeypatch):
    calls = []

    def fake_route(method, symbol, indicator, curr_date, look_back_days):
        calls.append((method, symbol, indicator, curr_date, look_back_days))
        return f"{indicator} result"

    monkeypatch.setattr(technical_indicators_tools, "route_to_vendor", fake_route)

    result = technical_indicators_tools.get_indicators.__wrapped__(
        symbol="AAPL",
        curr_date="2026-07-22",
        look_back_days=3,
    )

    assert tuple(inspect.signature(technical_indicators_tools.get_indicators).parameters) == (
        "symbol",
        "curr_date",
        "look_back_days",
    )
    assert technical_indicators_tools.MARKET_ANALYST_INDICATORS == EXPECTED_INDICATORS
    assert calls == [("get_indicators", "AAPL", indicator, "2026-07-22", 3) for indicator in EXPECTED_INDICATORS]
    assert result == "\n\n".join(f"{indicator} result" for indicator in EXPECTED_INDICATORS)


def test_get_indicators_reports_partial_failure_and_continues(monkeypatch):
    calls = []

    def fake_route(method, symbol, indicator, curr_date, look_back_days):
        calls.append(indicator)
        if indicator == "macd":
            raise RuntimeError("vendor unavailable")
        return f"{indicator} result"

    monkeypatch.setattr(technical_indicators_tools, "route_to_vendor", fake_route)

    result = technical_indicators_tools.get_indicators.__wrapped__(
        symbol="AAPL",
        curr_date="2026-07-22",
    )

    assert calls == list(EXPECTED_INDICATORS)
    assert "## macd unavailable\n\nError: vendor unavailable" in result
    assert "macds result" in result


def test_get_indicators_raises_when_every_indicator_fails(monkeypatch):
    def failing_route(method, symbol, indicator, curr_date, look_back_days):
        raise RuntimeError(f"{indicator} unavailable")

    monkeypatch.setattr(technical_indicators_tools, "route_to_vendor", failing_route)

    with pytest.raises(RuntimeError, match="Failed to retrieve all market indicators"):
        technical_indicators_tools.get_indicators.__wrapped__(
            symbol="AAPL",
            curr_date="2026-07-22",
        )


def test_indicator_routing_falls_back_after_error_string(monkeypatch):
    calls = []

    def alpha_vantage(*args):
        calls.append("alpha_vantage")
        return "Error: No data returned for rsi"

    def yfinance(*args):
        calls.append("yfinance")
        return "rsi result"

    monkeypatch.setattr(interface, "get_vendor", lambda category, method: "alpha_vantage")
    monkeypatch.setitem(
        interface.VENDOR_METHODS,
        "get_indicators",
        {"alpha_vantage": alpha_vantage, "yfinance": yfinance},
    )

    result = interface.route_to_vendor("get_indicators", "AAPL", "rsi", "2026-07-22", 30)

    assert result == "rsi result"
    assert calls == ["alpha_vantage", "yfinance"]


def test_indicator_routing_raises_when_all_vendors_return_errors(monkeypatch):
    def error_result(*args):
        return "Error: indicator unavailable"

    monkeypatch.setattr(interface, "get_vendor", lambda category, method: "alpha_vantage")
    monkeypatch.setitem(
        interface.VENDOR_METHODS,
        "get_indicators",
        {"alpha_vantage": error_result, "yfinance": error_result},
    )

    with pytest.raises(RuntimeError, match="All vendor implementations failed"):
        interface.route_to_vendor("get_indicators", "AAPL", "rsi", "2026-07-22", 30)


@patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"})
def test_market_analyst_prompt_requires_one_batch_indicator_call():
    from tradingscope.agents.analysts.market_analyst import create_market_analyst_agent
    from tradingscope.agents.utils.context import AgentContext

    context = AgentContext()
    context.company_of_interest = "AAPL"

    prompt = create_market_analyst_agent(context=context)._system_prompt

    assert "仅调用一次 `get_indicators`" in prompt
    assert all(indicator in prompt for indicator in EXPECTED_INDICATORS)
    assert "最多择 6 个" not in prompt
