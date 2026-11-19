"""Regression tests for the inclusive stock-data tool date window."""

from datetime import datetime, timezone
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from tradingscope.agents.utils.core_stock_tools import get_stock_data
from tradingscope.dataflows import interface, y_finance


@pytest.mark.parametrize(
    ("start_date", "end_date", "expected_dates"),
    [
        ("2026-11-16", "2026-11-17", ["2026-11-16", "2026-11-17"]),
        ("2026-11-17", "2026-11-17", ["2026-11-17"]),
    ],
)
def test_stock_data_tool_includes_the_requested_end_session(monkeypatch, start_date, end_date, expected_dates):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            fixed = datetime(2026, 11, 20, 22, tzinfo=timezone.utc)
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

    monkeypatch.setattr(y_finance, "datetime", Clock)
    data = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [101.0, 102.0, 103.0],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.DatetimeIndex(["2026-11-16", "2026-11-17", "2026-11-18"], name="Date"),
    )

    def history(*, start, end):
        return data.loc[(data.index >= start) & (data.index < end)].copy()

    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _symbol: SimpleNamespace(history=history))
    monkeypatch.setitem(interface.VENDOR_METHODS, "get_stock_data", {"yfinance": y_finance.get_YFin_data_online})

    result = get_stock_data("AAPL", start_date, end_date)

    text = "".join(block.text for block in result.content)
    assert "Date," in text
    returned = pd.read_csv(StringIO(text[text.index("Date,") :]))
    assert returned["Date"].tolist() == expected_dates
    assert f"from {start_date} to {end_date}" in text


@pytest.mark.parametrize(
    ("symbol", "hour_utc", "expected_dates"),
    [
        ("PRE", 13, ["2026-11-17"]),
        ("REG", 17, ["2026-11-17"]),
        ("POST", 22, ["2026-11-17", "2026-11-18"]),
    ],
)
def test_stock_data_excludes_unfinished_daily_bars_before_and_during_market(monkeypatch, symbol, hour_utc, expected_dates):
    from tradingscope.dataflows import stockstats_utils

    now = datetime(2026, 11, 18, hour_utc, tzinfo=timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    data = pd.DataFrame(
        {"Open": [100.0, 101.0], "High": [102.0, 103.0], "Low": [99.0, 100.0], "Close": [101.0, 102.0], "Volume": [1000, 1100]},
        index=pd.DatetimeIndex(["2026-11-17", "2026-11-18"], name="Date", tz="America/New_York"),
    )

    def history(*, start, end):
        dates = data.index.strftime("%Y-%m-%d")
        return data.loc[(dates >= start) & (dates < end)].copy()

    close = int(datetime(2026, 11, 18, 21, tzinfo=timezone.utc).timestamp())
    ticker = SimpleNamespace(history=history, history_metadata={"currentTradingPeriod": {"regular": {"end": close}}})
    monkeypatch.setattr(stockstats_utils, "datetime", Clock)
    monkeypatch.setattr(y_finance, "datetime", Clock)
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _symbol: ticker)
    monkeypatch.setitem(interface.VENDOR_METHODS, "get_stock_data", {"yfinance": y_finance.get_YFin_data_online})

    result = get_stock_data(symbol, "2026-11-17", "2026-11-18")

    text = "".join(block.text for block in result.content)
    returned = pd.read_csv(StringIO(text[text.index("Date,") :]))
    assert returned["Date"].tolist() == expected_dates


@pytest.mark.parametrize(
    ("symbol", "day", "close_hour_utc", "hour_utc", "expected_today", "market_timezone"),
    [
        ("EARLY", "2026-11-27", 18, 19, True, "America/New_York"),
        ("EARLYPRE", "2026-11-27", 18, 17, False, "America/New_York"),
        ("DST", "2026-07-24", 20, 21, True, "America/New_York"),
        ("0700.HK", "2026-11-18", 8, 9, True, "Asia/Hong_Kong"),
    ],
)
def test_stock_data_honors_exchange_close_and_daylight_saving(monkeypatch, symbol, day, close_hour_utc, hour_utc, expected_today, market_timezone):
    from tradingscope.dataflows import stockstats_utils

    now = datetime.fromisoformat(f"{day}T{hour_utc:02d}:00:00+00:00")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    previous_session = pd.Timestamp("2026-11-25") if day == "2026-11-27" else pd.Timestamp(day) - pd.Timedelta(days=1)
    dates = pd.DatetimeIndex([previous_session, pd.Timestamp(day)], name="Date", tz=market_timezone)
    data = pd.DataFrame({"Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 101.0, "Volume": 1000}, index=dates)
    close = int(datetime.fromisoformat(f"{day}T{close_hour_utc:02d}:00:00+00:00").timestamp())
    ticker = SimpleNamespace(
        history=lambda **_kwargs: data.copy(),
        history_metadata={"exchangeTimezoneName": market_timezone, "currentTradingPeriod": {"regular": {"end": close}}},
    )
    monkeypatch.setattr(stockstats_utils, "datetime", Clock)
    monkeypatch.setattr(y_finance, "datetime", Clock)
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _symbol: ticker)
    monkeypatch.setitem(interface.VENDOR_METHODS, "get_stock_data", {"yfinance": y_finance.get_YFin_data_online})

    result = get_stock_data(symbol, dates[0].strftime("%Y-%m-%d"), day)

    text = "".join(block.text for block in result.content)
    returned = pd.read_csv(StringIO(text[text.index("Date,") :]))
    assert (day in returned["Date"].tolist()) is expected_today


@pytest.mark.parametrize(
    ("now_iso", "close_iso", "market_timezone", "expected_today"),
    [
        ("2026-11-18T14:00:00+00:00", "2026-11-18T21:00:00+00:00", "America/New_York", False),
        ("2026-11-18T17:00:00+00:00", "2026-11-18T21:00:00+00:00", "America/New_York", False),
        ("2026-11-18T20:59:59+00:00", "2026-11-18T21:00:00+00:00", "America/New_York", False),
        ("2026-11-18T21:00:00+00:00", "2026-11-18T21:00:00+00:00", "America/New_York", True),
        ("2026-11-18T22:00:00+00:00", "2026-11-18T21:00:00+00:00", "America/New_York", True),
        ("2026-11-27T17:59:59+00:00", "2026-11-27T18:00:00+00:00", "America/New_York", False),
        ("2026-11-27T18:00:00+00:00", "2026-11-27T18:00:00+00:00", "America/New_York", True),
        ("2026-11-27T19:00:00+00:00", "2026-11-27T18:00:00+00:00", "America/New_York", True),
        ("2026-07-24T19:59:59+00:00", "2026-07-24T20:00:00+00:00", "America/New_York", False),
        ("2026-07-24T20:00:00+00:00", "2026-07-24T20:00:00+00:00", "America/New_York", True),
        ("2026-11-19T02:00:00+00:00", "2026-11-18T21:00:00+00:00", "America/New_York", True),
        ("2026-11-17T16:30:00+00:00", "2026-11-18T08:00:00+00:00", "Asia/Hong_Kong", False),
        ("2026-11-18T01:00:00+00:00", "2026-11-18T08:00:00+00:00", "Asia/Hong_Kong", False),
        ("2026-11-18T04:00:00+00:00", "2026-11-18T08:00:00+00:00", "Asia/Hong_Kong", False),
        ("2026-11-18T07:59:59+00:00", "2026-11-18T08:00:00+00:00", "Asia/Hong_Kong", False),
        ("2026-11-18T08:00:00+00:00", "2026-11-18T08:00:00+00:00", "Asia/Hong_Kong", True),
        ("2026-11-18T09:00:00+00:00", "2026-11-18T08:00:00+00:00", "Asia/Hong_Kong", True),
        ("2026-11-18T22:00:00+00:00", "2026-11-18T23:00:00+00:00", "Europe/London", False),
    ],
)
def test_stock_daily_session_boundaries(monkeypatch, request, now_iso, close_iso, market_timezone, expected_today):
    from tradingscope.dataflows import stockstats_utils

    now = datetime.fromisoformat(now_iso)
    close = datetime.fromisoformat(close_iso)
    day = pd.Timestamp(close).tz_convert(market_timezone).strftime("%Y-%m-%d")
    previous = "2026-11-25" if day == "2026-11-27" else (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    dates = pd.DatetimeIndex([previous, day], name="Date", tz=market_timezone)
    data = pd.DataFrame({"Close": [100, 101]}, index=dates)
    ticker = SimpleNamespace(
        history=lambda **_kwargs: data.copy(),
        history_metadata={"exchangeTimezoneName": market_timezone, "currentTradingPeriod": {"regular": {"end": int(close.timestamp())}}},
    )
    monkeypatch.setattr(stockstats_utils, "datetime", Clock)
    monkeypatch.setattr(y_finance, "datetime", Clock)
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _symbol: ticker)
    monkeypatch.setitem(interface.VENDOR_METHODS, "get_stock_data", {"yfinance": y_finance.get_YFin_data_online})

    result = get_stock_data(request.node.name, previous, day)
    text = "".join(block.text for block in result.content)
    returned = pd.read_csv(StringIO(text[text.index("Date,") :]))
    assert returned["Date"].tolist() == ([previous, day] if expected_today else [previous])


@pytest.mark.parametrize(
    ("market_timezone", "end_iso", "expected_today", "warning"),
    [
        ("America/New_York", None, False, "16:00"),
        ("America/New_York", "2026-11-25T21:00:00+00:00", False, "16:00"),
        ("America/New_York", "2026-11-30T21:00:00+00:00", False, "16:00"),
        ("America/New_York", "invalid", False, "16:00"),
        ("Asia/Hong_Kong", None, False, "Cannot confirm"),
        ("Asia/Hong_Kong", "2026-11-26T08:00:00+00:00", False, "Cannot confirm"),
    ],
)
def test_missing_or_stale_schedule_is_conservative(monkeypatch, request, caplog, market_timezone, end_iso, expected_today, warning):
    from tradingscope.dataflows import stockstats_utils

    now = datetime.fromisoformat("2026-11-27T19:00:00+00:00")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    day = pd.Timestamp(now).tz_convert(market_timezone).strftime("%Y-%m-%d")
    dates = pd.DatetimeIndex(["2026-11-25", day], name="Date", tz=market_timezone)
    data = pd.DataFrame({"Close": [100, 101]}, index=dates)
    end = int(datetime.fromisoformat(end_iso).timestamp()) if end_iso and end_iso != "invalid" else end_iso
    ticker = SimpleNamespace(history=lambda **_kwargs: data.copy(), history_metadata={"currentTradingPeriod": {"regular": {"end": end}}})
    monkeypatch.setattr(stockstats_utils, "datetime", Clock)
    monkeypatch.setattr(y_finance, "datetime", Clock)
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _symbol: ticker)
    monkeypatch.setitem(interface.VENDOR_METHODS, "get_stock_data", {"yfinance": y_finance.get_YFin_data_online})
    text = "".join(block.text for block in get_stock_data(request.node.name, "2026-11-25", day).content)
    returned = pd.read_csv(StringIO(text[text.index("Date,") :]))
    assert (day in returned["Date"].tolist()) is expected_today
    assert warning in caplog.text


def test_old_window_needs_no_current_schedule_and_filters_requested_dates(monkeypatch):
    data = pd.DataFrame({"Close": [99, 100, 101]}, index=pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"], name="Date"))
    ticker = SimpleNamespace(history=Mock(return_value=data))
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _symbol: ticker)
    text = y_finance.get_YFin_data_online("AAPL", "2020-01-02", "2020-01-02")
    returned = pd.read_csv(StringIO(text[text.index("Date,") :]))
    assert returned["Date"].tolist() == ["2020-01-02"]
    ticker.history.assert_called_once_with(start="2020-01-02", end="2020-01-03")


@pytest.mark.parametrize(
    ("now_iso", "previous", "end_date"),
    [
        ("2026-11-18T22:00:00+00:00", "2026-11-17", "2026-11-18"),
        ("2026-11-26T22:00:00+00:00", "2026-11-25", "2026-11-26"),
        ("2026-11-28T22:00:00+00:00", "2026-11-27", "2026-11-28"),
    ],
)
def test_stock_data_never_invents_a_missing_session(monkeypatch, request, now_iso, previous, end_date):
    now = datetime.fromisoformat(now_iso)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    prices = pd.DataFrame({"Close": [100]}, index=pd.DatetimeIndex([previous], name="Date", tz="America/New_York"))
    monkeypatch.setattr(y_finance, "datetime", Clock)
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _symbol: SimpleNamespace(history=lambda **_kwargs: prices))
    monkeypatch.setitem(interface.VENDOR_METHODS, "get_stock_data", {"yfinance": y_finance.get_YFin_data_online})
    text = "".join(block.text for block in get_stock_data(request.node.name, previous, end_date).content)
    returned = pd.read_csv(StringIO(text[text.index("Date,") :]))
    assert returned["Date"].tolist() == [previous]


def test_unknown_exchange_cannot_silently_use_new_york_close(monkeypatch, caplog):
    now = datetime.fromisoformat("2026-11-18T22:00:00+00:00")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    prices = pd.DataFrame({"Close": [100]}, index=pd.DatetimeIndex(["2026-11-18"], name="Date"))
    monkeypatch.setattr(y_finance, "datetime", Clock)
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _symbol: SimpleNamespace(history=lambda **_kwargs: prices, history_metadata={}))
    monkeypatch.setitem(interface.VENDOR_METHODS, "get_stock_data", {"yfinance": y_finance.get_YFin_data_online})
    with pytest.raises(RuntimeError, match="All vendor implementations failed"):
        get_stock_data("UNKNOWN-EXCHANGE", "2026-11-18", "2026-11-18")
    assert "Cannot determine exchange timezone" in caplog.text


@pytest.fixture
def return_history_api(monkeypatch):
    dates = pd.bdate_range(end="2026-11-17", periods=6)
    stock = pd.DataFrame({"Close": [100, 200, 105, 106, 108, 110], "Volume": [1000] * 6}, index=dates)
    sector = pd.DataFrame({"Close": [50, 55, 56, 58, 59, 60], "Volume": [1000] * 6}, index=dates)
    for frame in (stock, sector):
        frame["Open"] = frame["Close"]
        frame["High"] = frame["Close"] + 1
        frame["Low"] = frame["Close"] - 1
    tickers = {
        "XLK": SimpleNamespace(history=lambda **_kwargs: sector),
    }

    def ticker(symbol):
        return tickers.get(symbol, SimpleNamespace(info={"sector": "Technology", "industry": "Software"}, history=lambda **_kwargs: stock))

    monkeypatch.setattr(y_finance.yf, "Ticker", ticker)
    for name in ("get_sector_performance", "get_market_indices", "get_volume_analysis"):
        monkeypatch.setitem(interface.VENDOR_METHODS, name, {"yfinance": getattr(y_finance, name)})
    monkeypatch.setattr(interface, "get_vendor", lambda *_args: "yfinance")
    return stock, sector, tickers


def test_sector_index_and_volume_returns_use_five_close_intervals(return_history_api):
    from tradingscope.agents.utils import core_stock_tools

    stock, _, tickers = return_history_api
    earlier = stock.iloc[[0] * 14].copy()
    earlier.index = pd.bdate_range(end="2026-11-09", periods=14)
    history = pd.concat([earlier, stock])
    tickers["FIVE-INTERVALS"] = SimpleNamespace(info={"sector": "Technology"}, history=lambda **_kwargs: history)
    sector = "".join(block.text for block in core_stock_tools.get_sector_performance("FIVE-INTERVALS", 5).content)
    indices = "".join(block.text for block in core_stock_tools.get_market_indices(5).content)
    volume = "".join(block.text for block in core_stock_tools.get_volume_analysis("FIVE-INTERVALS", 5).content)
    assert "Stock 5-Day Return: +10.00% (2026-11-10 to 2026-11-17; 5 completed-session intervals)" in sector
    assert "Sector 5-Day Return: +20.00% (2026-11-10 to 2026-11-17; 5 completed-session intervals)" in sector
    assert "Stock vs Sector (5D): -10.00pp" in sector
    assert "5-Day Change: +10.00% (2026-11-10 to 2026-11-17; 5 completed-session intervals)" in indices
    assert "5-Day Price Change: +10.00% (2026-11-10 to 2026-11-17; 5 completed-session intervals)" in volume
    assert "5-Day Avg Volume: 1,000" in volume
    assert "OBV 5-Day Change: +3,000" in volume
    assert "5-Day Return: -45.00%" not in sector


def test_sector_relative_returns_do_not_compare_different_session_windows(return_history_api):
    from tradingscope.agents.utils import core_stock_tools

    _, sector, _ = return_history_api
    sector.index = pd.bdate_range(end="2026-11-16", periods=6)
    text = "".join(block.text for block in core_stock_tools.get_sector_performance("MISALIGNED-SESSIONS", 5).content)
    assert "Stock vs Sector (5D): N/A (session windows differ)" in text
    assert "Stock vs Sector (1D): N/A (session windows differ)" in text
    assert "Stock vs Sector (5D): -" not in text


def test_five_closes_are_insufficient_for_five_returns(return_history_api):
    from tradingscope.agents.utils import core_stock_tools

    stock, sector, _ = return_history_api
    stock.drop(stock.index[0], inplace=True)
    sector.drop(sector.index[0], inplace=True)
    text = "".join(block.text for block in core_stock_tools.get_sector_performance("FIVE-ROWS", 7).content)
    indices = "".join(block.text for block in core_stock_tools.get_market_indices(7).content)
    assert "Stock 5-Day Return: N/A (need 6 completed closes, got 5)" in text
    assert "Sector 5-Day Return: N/A (need 6 completed closes, got 5)" in text
    assert "5-Day Change: N/A (need 6 completed closes, got 5)" in indices
    assert "NASDAQ Recent Momentum" not in indices
    assert "Error -" not in indices


def test_ten_and_configured_returns_also_use_n_plus_one_closes(return_history_api):
    from tradingscope.agents.utils import core_stock_tools

    _, _, tickers = return_history_api
    dates = pd.bdate_range(end="2026-11-17", periods=11)
    hist = pd.DataFrame({"Close": [100, 150, 102, 103, 104, 105, 106, 107, 108, 109, 120]}, index=dates)
    for ticker in ("TEN-INTERVALS", "XLK", "^GSPC", "^IXIC", "^DJI", "^VIX", "^RUT", "^HXC", "MCHI", "^HSI"):
        tickers[ticker] = SimpleNamespace(info={"sector": "Technology"}, history=lambda **_kwargs: hist)
    sector = "".join(block.text for block in core_stock_tools.get_sector_performance("TEN-INTERVALS", 10).content)
    indices = "".join(block.text for block in core_stock_tools.get_market_indices(10).content)
    assert "Stock 10-Day Return: +20.00% (2026-11-03 to 2026-11-17; 10 completed-session intervals)" in sector
    assert "10-Day Change: +20.00% (2026-11-03 to 2026-11-17; 10 completed-session intervals)" in indices
    assert indices.count("10-Day Change:") == 8


def test_vix_spike_uses_five_intervals_not_five_samples(return_history_api):
    from tradingscope.agents.utils import core_stock_tools

    _, _, tickers = return_history_api
    hist = pd.DataFrame({"Close": [20, 25, 22, 23, 24, 26]}, index=pd.bdate_range(end="2026-11-17", periods=6))
    tickers["^VIX"] = SimpleNamespace(history=lambda **_kwargs: hist)
    text = "".join(block.text for block in core_stock_tools.get_market_indices(9).content)
    assert "VIX Spike Detected: 5-day surge +30.00%" in text
    assert "2026-11-10 to 2026-11-17" in text
