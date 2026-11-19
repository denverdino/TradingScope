import inspect
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
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


def test_market_analyst_prompt_requires_one_batch_indicator_call(mock_context):
    from tradingscope.agents.analysts.market_analyst import create_market_analyst_agent

    mock_context.ticker = "AAPL"

    prompt = create_market_analyst_agent(context=mock_context)._system_prompt

    assert "仅调用一次 `get_indicators`" in prompt
    assert all(indicator in prompt for indicator in EXPECTED_INDICATORS)
    assert "最多择 6 个" not in prompt


@pytest.fixture
def stale_price_cache(monkeypatch, tmp_path):
    from tradingscope.dataflows import config, stockstats_utils

    fixed_today = pd.Timestamp("2026-11-18")
    monkeypatch.setattr(pd.Timestamp, "today", Mock(return_value=fixed_today))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            fixed = datetime(2026, 11, 18, 13, tzinfo=timezone.utc)
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

    close = int(datetime(2026, 11, 18, 21, tzinfo=timezone.utc).timestamp())
    monkeypatch.setattr(stockstats_utils, "datetime", Clock)
    monkeypatch.setattr(
        stockstats_utils.yf,
        "Ticker",
        lambda _symbol: SimpleNamespace(history_metadata={"exchangeTimezoneName": "America/New_York", "currentTradingPeriod": {"regular": {"end": close}}}),
    )
    settings = config.get_config()
    settings["data_cache_dir"] = str(tmp_path)
    monkeypatch.setattr(config, "get_config", lambda: settings)
    monkeypatch.setattr(stockstats_utils, "get_config", lambda: settings)
    dates = pd.bdate_range(end="2026-11-17", periods=150, name="Date")
    old_data = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000}, index=dates)
    fresh_data = old_data.copy()
    fresh_data.iloc[-1, fresh_data.columns.get_loc("Close")] = 777.0
    fresh_data.iloc[-1, fresh_data.columns.get_loc("High")] = 800.0
    download = Mock(return_value=fresh_data)
    monkeypatch.setattr(stockstats_utils.yf, "download", download)

    def seed(symbol):
        for years in (2, 15):
            start_date = (fixed_today - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
            old_data.to_csv(tmp_path / f"{symbol}-YFin-data-{start_date}-2026-11-18.csv")
        return fresh_data, download

    return seed


def test_indicator_tool_refreshes_old_prices_once_for_the_whole_batch(stale_price_cache):
    _fresh_data, download = stale_price_cache("AAPL")

    first = technical_indicators_tools.get_indicators("AAPL", "2026-11-18", look_back_days=1)
    second = technical_indicators_tools.get_indicators("AAPL", "2026-11-18", look_back_days=2)

    for result in (first, second):
        text = "".join(block.text for block in result.content)
        latest_ema = float(text.split("2026-11-17:", 1)[1].splitlines()[0].strip())
        assert latest_ema == pytest.approx(223.090909, abs=0.01)
    download.assert_called_once()


def test_weekly_bollinger_tool_refreshes_old_prices(stale_price_cache):
    _fresh_data, download = stale_price_cache("MSFT")

    result = technical_indicators_tools.get_weekly_bollinger_signal("MSFT", "2026-11-18")

    text = "".join(block.text for block in result.content)
    assert "现价（周收盘）: 777.00" in text
    download.assert_called_once()


def test_indicator_fallback_reader_refreshes_old_prices(stale_price_cache):
    from tradingscope.dataflows.y_finance import get_stockstats_indicator

    _fresh_data, download = stale_price_cache("IBM")

    latest_ema = get_stockstats_indicator("IBM", "close_10_ema", "2026-11-17")

    assert float(latest_ema) == pytest.approx(223.090909, abs=0.01)
    download.assert_called_once()


def test_empty_refresh_does_not_reuse_stale_data_or_block_a_retry(stale_price_cache):
    fresh_data, download = stale_price_cache("EMPTY")
    download.return_value = pd.DataFrame()

    with pytest.raises(RuntimeError, match="No price data"):
        technical_indicators_tools.get_weekly_bollinger_signal("EMPTY", "2026-11-18")

    download.return_value = fresh_data
    result = technical_indicators_tools.get_weekly_bollinger_signal("EMPTY", "2026-11-18")

    assert "现价（周收盘）: 777.00" in "".join(block.text for block in result.content)
    assert download.call_count == 2


def test_missing_daily_bar_is_not_mislabeled_as_a_market_holiday(stale_price_cache):
    from tradingscope.dataflows.y_finance import get_stockstats_indicator

    stale_price_cache("MISSING")

    result = technical_indicators_tools.get_indicators("MISSING", "2026-11-18", look_back_days=1)
    text = "".join(block.text for block in result.content)

    assert "2026-11-18: N/A: No price data available for this date" in text
    assert "weekend or holiday" not in text
    assert get_stockstats_indicator("MISSING", "close_10_ema", "2026-11-18") == "N/A: No price data available for this date"


@pytest.mark.parametrize(("hour_utc", "expected_close", "today_completed"), [(13, 777.0, False), (17, 777.0, False), (22, 888.0, True)])
def test_indicator_tools_use_only_completed_bars_in_each_trading_phase(stale_price_cache, monkeypatch, hour_utc, expected_close, today_completed):
    from tradingscope.dataflows import stockstats_utils

    symbol = f"PHASE{hour_utc}"
    fresh_data, download = stale_price_cache(symbol)
    today_bar = fresh_data.iloc[-1:].copy()
    today_bar.index = pd.DatetimeIndex(["2026-11-18"], name="Date")
    today_bar["Close"] = 888.0
    today_bar["High"] = 900.0
    download.return_value = pd.concat([fresh_data, today_bar])

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            fixed = datetime(2026, 11, 18, hour_utc, tzinfo=timezone.utc)
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

    monkeypatch.setattr(stockstats_utils, "datetime", Clock)

    indicators = technical_indicators_tools.get_indicators(symbol, "2026-11-18", 1)
    weekly = technical_indicators_tools.get_weekly_bollinger_signal(symbol, "2026-11-18")

    text = "".join(block.text for block in indicators.content)
    today_value = text.split("2026-11-18: ", 1)[1].splitlines()[0].strip()
    assert ("N/A" not in today_value) is today_completed
    assert f"现价（周收盘）: {expected_close:.2f}" in "".join(block.text for block in weekly.content)
    assert download.call_count == 2


@pytest.mark.parametrize(
    ("now_iso", "day", "previous", "close_iso", "market_timezone", "complete"),
    [
        ("2026-11-27T17:59:59+00:00", "2026-11-27", "2026-11-25", "2026-11-27T18:00:00+00:00", "America/New_York", False),
        ("2026-11-27T18:00:00+00:00", "2026-11-27", "2026-11-25", "2026-11-27T18:00:00+00:00", "America/New_York", True),
        ("2026-07-24T19:59:59+00:00", "2026-07-24", "2026-07-23", "2026-07-24T20:00:00+00:00", "America/New_York", False),
        ("2026-07-24T20:00:00+00:00", "2026-07-24", "2026-07-23", "2026-07-24T20:00:00+00:00", "America/New_York", True),
        ("2026-11-17T16:30:00+00:00", "2026-11-18", "2026-11-17", "2026-11-18T08:00:00+00:00", "Asia/Hong_Kong", False),
        ("2026-11-18T07:59:59+00:00", "2026-11-18", "2026-11-17", "2026-11-18T08:00:00+00:00", "Asia/Hong_Kong", False),
        ("2026-11-18T08:00:00+00:00", "2026-11-18", "2026-11-17", "2026-11-18T08:00:00+00:00", "Asia/Hong_Kong", True),
    ],
)
def test_indicators_and_weekly_share_exchange_session_boundaries(monkeypatch, tmp_path, request, now_iso, day, previous, close_iso, market_timezone, complete):
    from tradingscope.dataflows import config, stockstats_utils, y_finance

    now = datetime.fromisoformat(now_iso)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    symbol = f"EDGE{hash(request.node.nodeid)}"
    dates = pd.bdate_range(end=previous, periods=150, name="Date", tz=market_timezone)
    prices = pd.DataFrame({"Open": 100.0, "High": 110.0, "Low": 90.0, "Close": 100.0, "Volume": 1000}, index=dates)
    today = prices.iloc[-1:].copy()
    today.index = pd.DatetimeIndex([day], name="Date", tz=market_timezone)
    today["Close"] = 888.0
    today["High"] = 900.0
    download = Mock(return_value=pd.concat([prices, today]))
    metadata = {"exchangeTimezoneName": market_timezone, "currentTradingPeriod": {"regular": {"end": int(datetime.fromisoformat(close_iso).timestamp())}}}
    monkeypatch.setattr(stockstats_utils, "datetime", Clock)
    monkeypatch.setattr(stockstats_utils.yf, "Ticker", lambda _symbol: SimpleNamespace(history_metadata=metadata))
    monkeypatch.setattr(stockstats_utils.yf, "download", download)
    settings = config.get_config()
    settings["data_cache_dir"] = str(tmp_path)
    monkeypatch.setattr(config, "get_config", lambda: settings)
    monkeypatch.setattr(interface, "get_vendor", lambda *_args: "yfinance")
    monkeypatch.setitem(interface.VENDOR_METHODS, "get_indicators", {"yfinance": y_finance.get_stock_stats_indicators_window})

    indicators = technical_indicators_tools.get_indicators(symbol, day, 1)
    weekly = technical_indicators_tools.get_weekly_bollinger_signal(symbol, day)
    indicator_text = "".join(block.text for block in indicators.content)
    today_value = indicator_text.split(f"{day}: ", 1)[1].splitlines()[0]
    assert ("N/A" not in today_value) is complete
    assert f"现价（周收盘）: {888.0 if complete else 100.0:.2f}" in "".join(block.text for block in weekly.content)
    assert download.call_count == 2


def test_failed_download_does_not_reuse_old_cache_and_can_be_retried(stale_price_cache):
    fresh_data, download = stale_price_cache("FAILED-DOWNLOAD")
    download.side_effect = ConnectionError("Yahoo unavailable")
    with pytest.raises(ConnectionError, match="Yahoo unavailable"):
        technical_indicators_tools.get_weekly_bollinger_signal("FAILED-DOWNLOAD", "2026-11-18")
    download.side_effect = None
    download.return_value = fresh_data
    result = technical_indicators_tools.get_weekly_bollinger_signal("FAILED-DOWNLOAD", "2026-11-18")
    assert "现价（周收盘）: 777.00" in "".join(block.text for block in result.content)
    assert download.call_count == 2


def test_indicator_readers_use_index_timezone_when_metadata_is_missing(stale_price_cache, monkeypatch):
    from tradingscope.dataflows import stockstats_utils, y_finance

    fresh_data, download = stale_price_cache("INDEX-TIMEZONE")
    download.return_value = fresh_data.tz_localize("America/New_York")
    monkeypatch.setattr(stockstats_utils.yf, "Ticker", lambda _symbol: SimpleNamespace(history_metadata={}))
    weekly = technical_indicators_tools.get_weekly_bollinger_signal("INDEX-TIMEZONE", "2026-11-18")
    assert "现价（周收盘）: 777.00" in "".join(block.text for block in weekly.content)
    assert float(y_finance.get_stockstats_indicator("INDEX-TIMEZONE", "close_10_ema", "2026-11-17")) == pytest.approx(223.090909, abs=0.01)
    assert float(y_finance.get_stockstats_indicator("INDEX-TIMEZONE", "rsi", "2026-11-17")) == 100.0
    assert download.call_count == 2


def test_download_with_only_unfinished_rows_is_not_cached(stale_price_cache):
    fresh_data, download = stale_price_cache("ONLY-UNFINISHED")
    unfinished = fresh_data.iloc[-1:].copy()
    unfinished.index = pd.DatetimeIndex(["2026-11-18"], name="Date")
    download.return_value = unfinished
    with pytest.raises(RuntimeError, match="No price data"):
        technical_indicators_tools.get_weekly_bollinger_signal("ONLY-UNFINISHED", "2026-11-18")
    download.return_value = fresh_data
    result = technical_indicators_tools.get_weekly_bollinger_signal("ONLY-UNFINISHED", "2026-11-18")
    assert "现价（周收盘）: 777.00" in "".join(block.text for block in result.content)
    assert download.call_count == 2
