from datetime import date
from unittest.mock import Mock, patch

import pandas as pd

from tradingscope.dataflows import y_finance


def _frame(stamps):
    return pd.DataFrame({"EPS Estimate": [1.0] * len(stamps)}, index=pd.DatetimeIndex(stamps))


def test_benchmark_uses_sector_mapping_and_adr_fallback():
    with patch.object(y_finance.yf, "Ticker", return_value=Mock(info={"sector": "Technology"})) as ticker:
        assert y_finance.get_sector_benchmark_symbol("AAPL") == "XLK"
        assert y_finance.get_sector_benchmark_symbol("BABA") == "SPY"
    ticker.assert_called_once_with("AAPL")
    with patch.object(y_finance.yf, "Ticker", return_value=Mock(info={"sector": "Unknown"})):
        assert y_finance.get_sector_benchmark_symbol("ABC") == "SPY"


def test_historical_earnings_dates_use_new_york_calendar():
    frame = _frame(["2025-01-01 00:00:00+00:00", "2026-07-15 00:30:00+00:00", "2026-10-01 20:00:00+00:00"])
    with patch.object(y_finance, "_earnings_dates_frame", return_value=frame):
        releases = y_finance.fetch_earnings_dates("AAPL", "2026-07-01", "2026-07-31")
    assert len(releases) == 1
    assert releases[0].date() == date(2026, 7, 14)
    assert releases[0].hour == 20
    assert releases[0].tzinfo is not None


def test_missing_earnings_is_unknown_not_a_confirmed_empty_calendar():
    with patch.object(y_finance, "_earnings_dates_frame", return_value=None):
        assert y_finance.fetch_earnings_dates("AAPL", "2026-07-01", "2026-07-31") is None
        assert "No earnings calendar available" in y_finance.get_earnings_calendar("AAPL", "2026-07-01")


def test_calendar_reports_next_release_without_reporting_future_eps():
    frame = _frame(["2026-07-01 16:00:00-04:00", "2026-07-20 16:00:00-04:00", "2026-08-20 16:00:00-04:00"])
    with patch.object(y_finance, "_earnings_dates_frame", return_value=frame):
        text = y_finance.get_earnings_calendar("AAPL", "2026-07-14")
    assert "Next earnings release: 2026-07-20 16:00" in text
    assert "Past release: 2026-07-01" in text
    assert "EPS" not in text
    assert "2026-08-20" not in text


def test_calendar_is_registered_in_routing():
    from tradingscope.dataflows.interface import get_category_for_method, route_to_vendor

    assert get_category_for_method("get_earnings_calendar") == "fundamental_data"
    with patch.dict("tradingscope.dataflows.interface.VENDOR_METHODS", {"get_earnings_calendar": {"yfinance": lambda *_args: "calendar"}}):
        assert route_to_vendor("get_earnings_calendar", "AAPL", "2026-07-14") == "calendar"
