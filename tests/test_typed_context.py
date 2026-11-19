"""Tests for typed AgentContext report state."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from tests.test_output_models import _all_outputs
from tradingscope.agents.renderers import render_evidence_pack
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.dashscope_response_model import DashScopeResponseModel
from tradingscope.agents.utils.prompt_cache import CACHE_PREFIX_BREAK, DashScopeResponseFormatter
from tradingscope.default_config import DEFAULT_CONFIG


def _typed_context() -> AgentContext:
    market, fundamentals, news, social, research, trader, portfolio = _all_outputs()
    context = AgentContext.__new__(AgentContext)
    context.ticker = "AAPL"
    context.analysis_date = "2026-07-14"
    context.last_completed_session = "2026-07-13"
    context.market = market
    context.fundamentals = fundamentals
    context.news = news
    context.social_media = social
    context.research_manager = research
    context.trader = trader
    context.portfolio_manager = portfolio
    return context


def test_context_renders_analysts_from_typed_outputs() -> None:
    context = _typed_context()

    markdown = context.generate_analyst_reports_md()

    assert context.market.decision.summary in markdown
    assert context.fundamentals.decision.summary in markdown
    assert context.news.decision.summary in markdown
    assert context.social_media.decision.summary in markdown


def test_context_renders_typed_downstream_inputs() -> None:
    context = _typed_context()

    trader_context = context.generate_trader_context_md()
    risk_context = context.generate_risk_evaluation_context_md()

    assert context.research_manager.decision.summary in trader_context
    assert context.trader.decision.summary in risk_context


def test_downstream_contexts_extend_a_stable_prefix() -> None:
    context = _typed_context()

    analysts = context.generate_analyst_reports_md()
    research = render_evidence_pack(context.research_manager)
    trader = render_evidence_pack(context.trader)

    assert context.generate_trader_context_md().split(CACHE_PREFIX_BREAK) == [analysts, research]
    assert context.generate_risk_evaluation_context_md().split(CACHE_PREFIX_BREAK) == [analysts, research, trader]


def test_default_model_config() -> None:
    assert DEFAULT_CONFIG["deep_think_llm"] == "qwen4-max"
    assert DEFAULT_CONFIG["quick_think_llm"] == "qwen3.8-flash"


def test_context_uses_two_responses_models() -> None:
    with (
        patch("tradingscope.agents.utils.context.get_last_completed_us_session", return_value="2026-08-10"),
        patch("tradingscope.agents.utils.context.OpenAICredential") as credential_factory,
        patch("tradingscope.agents.utils.context.DashScopeResponseModel") as response_model,
    ):
        response_model.Parameters = DashScopeResponseModel.Parameters
        AgentContext()

    assert response_model.call_count == 2
    assert response_model.call_args_list[0].kwargs["model"] == DEFAULT_CONFIG["deep_think_llm"]
    assert response_model.call_args_list[0].kwargs["parameters"].thinking_enable is True
    assert response_model.call_args_list[1].kwargs["parameters"].thinking_enable is False
    assert all(isinstance(call.kwargs["formatter"], DashScopeResponseFormatter) for call in response_model.call_args_list)
    assert response_model.call_args_list[0].kwargs["formatter"] is response_model.call_args_list[1].kwargs["formatter"]
    assert all(call.kwargs["client_kwargs"] == {"default_headers": {"x-dashscope-session-cache": "enable"}} for call in response_model.call_args_list)
    assert all(call.kwargs["stream"] is True for call in response_model.call_args_list)
    assert all(call.kwargs["credential"] is credential_factory.return_value for call in response_model.call_args_list)


def test_context_configures_responses_credential() -> None:
    with (
        patch("tradingscope.agents.utils.context.get_last_completed_us_session", return_value="2026-08-10"),
        patch("tradingscope.agents.utils.context.OpenAICredential") as credential_factory,
        patch("tradingscope.agents.utils.context.DashScopeResponseModel"),
    ):
        AgentContext()

    assert credential_factory.call_args.kwargs["base_url"] == DEFAULT_CONFIG["backend_url"]


@pytest.mark.parametrize(
    ("now_iso", "expected_session"),
    [
        ("2026-11-18T13:00:00+00:00", "2026-11-17"),
        ("2026-11-18T17:00:00+00:00", "2026-11-17"),
        ("2026-11-18T22:00:00+00:00", "2026-11-18"),
    ],
)
def test_context_session_excludes_today_before_and_during_trading(monkeypatch, now_iso, expected_session):
    from tradingscope.agents.utils import context
    from tradingscope.dataflows import stockstats_utils

    now = datetime.fromisoformat(now_iso)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    history = pd.DataFrame({"Close": [100, 101]}, index=pd.DatetimeIndex(["2026-11-17", "2026-11-18"], tz="America/New_York"))
    close = int(datetime.fromisoformat("2026-11-18T21:00:00+00:00").timestamp())
    ticker = SimpleNamespace(history=lambda **_kwargs: history, history_metadata={"currentTradingPeriod": {"regular": {"end": close}}})
    monkeypatch.setattr(stockstats_utils, "datetime", Clock)
    monkeypatch.setattr(context.yf, "Ticker", lambda _symbol: ticker)

    assert context.get_last_completed_us_session() == expected_session


def test_context_session_unavailable_does_not_guess_a_weekday(monkeypatch):
    from tradingscope.agents.utils import context
    from tradingscope.dataflows import stockstats_utils

    now = datetime.fromisoformat("2026-11-18T04:00:00+00:00")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    def unavailable(_symbol):
        raise RuntimeError("Yahoo unavailable")

    monkeypatch.setattr(stockstats_utils, "datetime", Clock)
    monkeypatch.setattr(context, "datetime", Clock)
    monkeypatch.setattr(context.yf, "Ticker", unavailable)

    with pytest.raises(RuntimeError, match="Cannot determine last completed US session"):
        context.get_last_completed_us_session()


@pytest.mark.parametrize(
    ("now_iso", "close_iso", "sessions", "expected"),
    [
        ("2026-11-27T17:59:59+00:00", "2026-11-27T18:00:00+00:00", ["2026-11-25", "2026-11-27"], "2026-11-25"),
        ("2026-11-27T18:00:00+00:00", "2026-11-27T18:00:00+00:00", ["2026-11-25", "2026-11-27"], "2026-11-27"),
        ("2026-11-27T19:00:00+00:00", "2026-11-27T18:00:00+00:00", ["2026-11-25", "2026-11-27"], "2026-11-27"),
        ("2026-07-24T19:59:59+00:00", "2026-07-24T20:00:00+00:00", ["2026-07-23", "2026-07-24"], "2026-07-23"),
        ("2026-07-24T20:00:00+00:00", "2026-07-24T20:00:00+00:00", ["2026-07-23", "2026-07-24"], "2026-07-24"),
        ("2026-11-18T20:59:59+00:00", "2026-11-18T21:00:00+00:00", ["2026-11-17", "2026-11-18"], "2026-11-17"),
        ("2026-11-18T21:00:00+00:00", "2026-11-18T21:00:00+00:00", ["2026-11-17", "2026-11-18"], "2026-11-18"),
        ("2026-11-19T02:00:00+00:00", "2026-11-18T21:00:00+00:00", ["2026-11-17", "2026-11-18"], "2026-11-18"),
        ("2026-11-18T22:00:00+00:00", "2026-11-18T21:00:00+00:00", ["2026-11-17"], "2026-11-17"),
        ("2026-11-26T22:00:00+00:00", "2026-11-27T18:00:00+00:00", ["2026-11-24", "2026-11-25"], "2026-11-25"),
        ("2026-11-28T22:00:00+00:00", "2026-11-27T18:00:00+00:00", ["2026-11-25", "2026-11-27"], "2026-11-27"),
    ],
)
def test_context_last_completed_session_requires_an_actual_row(monkeypatch, now_iso, close_iso, sessions, expected):
    from tradingscope.agents.utils import context
    from tradingscope.dataflows import stockstats_utils

    now = datetime.fromisoformat(now_iso)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    history = pd.DataFrame({"Close": 100}, index=pd.DatetimeIndex(sessions, tz="America/New_York"))
    ticker = SimpleNamespace(
        history=lambda **_kwargs: history, history_metadata={"currentTradingPeriod": {"regular": {"end": int(datetime.fromisoformat(close_iso).timestamp())}}}
    )
    monkeypatch.setattr(stockstats_utils, "datetime", Clock)
    monkeypatch.setattr(context.yf, "Ticker", lambda _symbol: ticker)
    assert context.get_last_completed_us_session() == expected


@pytest.mark.parametrize("sessions", [[], ["2099-01-01"]])
def test_context_rejects_empty_or_only_unfinished_history(monkeypatch, sessions):
    from tradingscope.agents.utils import context

    ticker = SimpleNamespace(history=lambda **_kwargs: pd.DataFrame({"Close": 100}, index=pd.DatetimeIndex(sessions)), history_metadata={})
    monkeypatch.setattr(context.yf, "Ticker", lambda _symbol: ticker)
    with pytest.raises(RuntimeError, match="Cannot determine last completed US session"):
        context.get_last_completed_us_session()
