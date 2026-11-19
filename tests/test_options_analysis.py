"""Options regressions through the real core tool and vendor route."""

from collections import namedtuple
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tradingscope.agents.utils.core_stock_tools import get_options_analysis
from tradingscope.dataflows import y_finance
from tradingscope.dataflows.interface import TOOLS_CATEGORIES, VENDOR_METHODS, get_category_for_method


def _make_options_chain(calls_data, puts_data):
    OptionsChain = namedtuple("OptionsChain", ["calls", "puts"])
    return OptionsChain(calls=pd.DataFrame(calls_data), puts=pd.DataFrame(puts_data))


def _sample_calls():
    return {
        "strike": [180.0, 190.0, 200.0],
        "volume": [1500, 3000, 5000],
        "openInterest": [10000, 25000, 50000],
        "impliedVolatility": [0.35, 0.30, 0.28],
    }


def _sample_puts():
    return {
        "strike": [170.0, 180.0, 190.0],
        "volume": [2000, 4000, 1000],
        "openInterest": [30000, 45000, 8000],
        "impliedVolatility": [0.32, 0.29, 0.27],
    }


@pytest.fixture
def options_api(monkeypatch, request):
    ticker = MagicMock()
    ticker.options = ("2026-04-24", "2026-05-01")
    chains = {expiry: _make_options_chain(_sample_calls(), _sample_puts()) for expiry in ticker.options}
    ticker.option_chain.side_effect = lambda expiry: chains[expiry]
    ticker.info = {"currentPrice": 192.50}
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _symbol: ticker)

    def freeze(utc_time):
        instant = datetime.fromisoformat(utc_time)

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

        monkeypatch.setattr(y_finance, "datetime", Clock)

    freeze("2026-04-23T16:00:00+00:00")

    def analyze():
        # Unique public inputs avoid the production tool cache, without clearing internals.
        chunk = get_options_analysis(f"AAPL-{request.node.name}")
        return "".join(block.text for block in chunk.content)

    return ticker, chains, freeze, analyze


class TestOptionsDataLayer:
    def test_us_stock_options(self, options_api):
        _, _, _, analyze = options_api
        result = analyze()
        assert "Options Chain Analysis for AAPL" in result
        assert "Put/Call Ratio" in result
        assert "Support Levels" in result
        assert "Resistance Levels" in result
        assert "Max Pain Price: $180.00" in result
        assert "not a high-probability price predictor" in result

    def test_no_options_data(self, options_api):
        ticker, _, _, analyze = options_api
        ticker.options = ()
        assert "No options data available" in analyze()
        ticker.option_chain.assert_not_called()

    def test_api_error(self, options_api):
        ticker, _, _, analyze = options_api
        ticker.option_chain.side_effect = ConnectionError("Network error")
        result = analyze()
        assert "Error retrieving options analysis" in result
        assert "Network error" in result

    def test_pcr_calculation(self, options_api):
        _, _, _, analyze = options_api
        result = analyze()
        assert "PCR (Open Interest): 0.98" in result
        assert "PCR (Volume): 0.74" in result
        assert "中性偏多" in result
        assert "Total Call Open Interest: 85,000" in result
        assert "Total Put Open Interest: 83,000" in result

    def test_zero_call_oi(self, options_api):
        ticker, chains, _, analyze = options_api
        calls = _sample_calls()
        calls["openInterest"] = [0, 0, 0]
        ticker.options = ("2026-04-24",)
        chains["2026-04-24"] = _make_options_chain(calls, _sample_puts())
        result = analyze()
        assert "PCR (Open Interest): N/A (Call OI = 0)" in result
        assert "Max Pain Price: N/A (non-unique minimum)" in result
        assert "Max Pain Minimum Interval: $190.00–unbounded" in result

    def test_support_resistance_ordering(self, options_api):
        _, _, _, analyze = options_api
        result = analyze()
        support = result.split("## Support Levels", 1)[1].split("## Resistance Levels", 1)[0]
        resistance = result.split("## Resistance Levels", 1)[1].split("## Max Pain", 1)[0]
        assert support.index("180.00") < support.index("170.00")
        assert resistance.index("200.00") < resistance.index("190.00")


@pytest.mark.parametrize(
    ("utc_time", "today", "future", "days"),
    [
        ("2026-11-20T14:00:00+00:00", "2026-11-20", "2026-11-23", 3),  # pre-market
        ("2026-11-20T17:00:00+00:00", "2026-11-20", "2026-11-23", 3),  # intraday
        ("2026-11-20T20:59:59+00:00", "2026-11-20", "2026-11-23", 3),
        ("2026-11-20T21:00:00+00:00", "2026-11-20", "2026-11-23", 3),  # winter close
        ("2026-11-20T22:00:00+00:00", "2026-11-20", "2026-11-23", 3),  # post-market
        ("2026-11-27T17:59:59+00:00", "2026-11-27", "2026-11-30", 3),
        ("2026-11-27T18:00:00+00:00", "2026-11-27", "2026-11-30", 3),  # early close
        ("2026-11-27T19:00:00+00:00", "2026-11-27", "2026-11-30", 3),
        ("2026-07-24T20:00:00+00:00", "2026-07-24", "2026-07-27", 3),  # DST close
        ("2026-11-21T01:00:00+00:00", "2026-11-20", "2026-11-21", 1),  # UTC crosses date
        ("2026-07-25T02:00:00+00:00", "2026-07-24", "2026-07-25", 1),
        ("2026-03-08T06:30:00+00:00", "2026-03-08", "2026-03-09", 1),  # DST boundary
        ("2026-03-08T07:30:00+00:00", "2026-03-08", "2026-03-09", 1),
    ],
)
def test_only_strictly_future_new_york_expirations_are_selected(options_api, utc_time, today, future, days):
    ticker, chains, freeze, analyze = options_api
    freeze(utc_time)
    ticker.options = (today, "2027-01-15", "2026-01-01", future)
    chains[today] = _make_options_chain(_sample_calls(), _sample_puts())
    chains[future] = _make_options_chain(_sample_calls(), _sample_puts())
    result = analyze()
    assert f"Selected Expiration: {future} ({days} calendar days away)" in result
    assert [call.args[0] for call in ticker.option_chain.call_args_list] == [future]
    assert "Nearest Expiration" not in result


def test_zero_oi_chain_is_skipped_for_next_complete_positive_chain(options_api):
    ticker, chains, _, analyze = options_api
    calls, puts = _sample_calls(), _sample_puts()
    calls["openInterest"] = puts["openInterest"] = [0, 0, 0]
    chains["2026-04-24"] = _make_options_chain(calls, puts)
    ticker.options = ("2026-05-01", "2026-04-24")
    result = analyze()
    assert "Selected Expiration: 2026-05-01 (8 calendar days away)" in result
    assert "2026-04-24: combined OI = 0" in result
    assert [call.args[0] for call in ticker.option_chain.call_args_list] == ["2026-04-24", "2026-05-01"]


@pytest.mark.parametrize("invalid", ["missing", "empty", None, float("nan"), "invalid", -1, 0.5, float("inf")])
def test_incomplete_put_oi_remains_unknown_but_call_references_are_usable(options_api, invalid):
    ticker, chains, _, analyze = options_api
    ticker.options = ("2026-04-24",)
    puts = _sample_puts()
    if invalid == "missing":
        del puts["openInterest"]
    elif invalid == "empty":
        puts = {}
    else:
        puts["openInterest"][0] = invalid
    chains["2026-04-24"] = _make_options_chain(_sample_calls(), puts)
    result = analyze()
    assert "Selected Expiration: 2026-04-24" in result
    assert "Limited report" in result
    assert "Total Put Open Interest: N/A" in result
    assert "Total Call Open Interest: 85,000" in result
    assert "PCR (Open Interest): N/A" in result
    assert "Max Pain Price: N/A" in result
    assert "Max Pain Price: $" not in result
    assert "Sentiment: N/A" in result
    support = result.split("## Support Levels", 1)[1].split("## Resistance Levels", 1)[0]
    resistance = result.split("## Resistance Levels", 1)[1].split("## Max Pain", 1)[0]
    assert "N/A" in support and "| 1 |" not in support
    assert "| 1 | $200.00 | 50,000 |" in resistance
    if invalid != "empty":
        assert "PCR (Volume): 0.74" in result
    else:
        assert "PCR (Volume): N/A" in result


@pytest.mark.parametrize("bad_column", ["openInterest", "strike"])
def test_no_numeric_max_pain_with_zero_or_missing_inputs(options_api, bad_column):
    ticker, chains, _, analyze = options_api
    ticker.options = ("2026-04-24",)
    calls, puts = _sample_calls(), _sample_puts()
    if bad_column == "openInterest":
        calls[bad_column] = puts[bad_column] = [0, 0, 0]
    else:
        del puts[bad_column]
    chains["2026-04-24"] = _make_options_chain(calls, puts)
    result = analyze()
    assert "Max Pain Price: N/A" in result
    assert "Max Pain Price: $" not in result
    if bad_column == "openInterest":
        assert "PCR (Open Interest): N/A" in result
        assert "Sentiment: N/A" in result
        assert "Total Call Open Interest: 0" in result
        assert "Limited report" in result
    assert "OI update time: unknown" in result
    assert "not verified complete current-session volume" in result
    assert "2026-04-23T12:00:00-04:00 (America/New_York)" in result


def test_only_volume_is_usable_when_both_oi_columns_are_absent(options_api):
    ticker, chains, _, analyze = options_api
    ticker.options = ("2026-04-24",)
    calls, puts = _sample_calls(), _sample_puts()
    del calls["openInterest"]
    del puts["openInterest"]
    chains["2026-04-24"] = _make_options_chain(calls, puts)
    result = analyze()
    assert "PCR (Volume): 0.74" in result
    assert "Total Call Open Interest: N/A" in result
    assert "Total Put Open Interest: N/A" in result
    assert "Max Pain Price: $" not in result
    assert "| 1 |" not in result
    assert "Sentiment: N/A" in result


@pytest.mark.parametrize("invalid", ["missing", None, float("nan"), "invalid", -1, 0.5, float("inf")])
def test_missing_put_volume_is_not_silently_zero(options_api, invalid):
    ticker, chains, _, analyze = options_api
    ticker.options = ("2026-04-24",)
    puts = _sample_puts()
    if invalid == "missing":
        del puts["volume"]
    else:
        puts["volume"][0] = invalid
    chains["2026-04-24"] = _make_options_chain(_sample_calls(), puts)
    result = analyze()
    assert "PCR (Volume): N/A" in result
    assert "PCR (Open Interest): 0.98" in result
    assert "Max Pain Price: $180.00" in result


def test_true_zero_put_oi_and_volume_are_valid_numerators(options_api):
    ticker, chains, _, analyze = options_api
    ticker.options = ("2026-04-24",)
    puts = _sample_puts()
    puts["openInterest"] = puts["volume"] = [0, 0, 0]
    chains["2026-04-24"] = _make_options_chain(_sample_calls(), puts)
    result = analyze()
    assert "PCR (Open Interest): 0.00" in result
    assert "PCR (Volume): 0.00" in result
    assert "Max Pain Price: N/A (non-unique minimum)" in result
    assert "Max Pain Minimum Interval: $0.00–$180.00" in result
    assert "Limited report" not in result


@pytest.mark.parametrize("reverse_rows", [False, True])
def test_sparse_oi_flat_minimum_is_an_interval_not_a_single_price(options_api, reverse_rows):
    ticker, chains, _, analyze = options_api
    ticker.options = ("2026-04-24",)
    calls = {"strike": [155.0, 130.0, 560.0], "openInterest": [47, 0, 121], "volume": [1, 0, 2]}
    puts = {"strike": [120.0, 140.0, 350.0], "openInterest": [147, 0, 0], "volume": [3, 0, 0]}
    if reverse_rows:
        calls = {key: values[::-1] for key, values in calls.items()}
        puts = {key: values[::-1] for key, values in puts.items()}
    chains["2026-04-24"] = _make_options_chain(calls, puts)
    result = analyze()
    assert "Max Pain Price: N/A (non-unique minimum)" in result
    assert "Max Pain Minimum Interval: $120.00–$155.00" in result
    assert "Max Pain Price: $" not in result
    assert "Distance from current price:" not in result
    assert "PCR (Open Interest): 0.88" in result
    assert "not a high-probability price predictor" in result


def test_equal_call_and_put_oi_can_have_a_positive_flat_minimum(options_api):
    ticker, chains, _, analyze = options_api
    ticker.options = ("2026-04-24",)
    calls = {"strike": [100.0, 115.0], "openInterest": [10, 0], "volume": [1, 0]}
    puts = {"strike": [120.0, 110.0], "openInterest": [10, 0], "volume": [1, 0]}
    chains["2026-04-24"] = _make_options_chain(calls, puts)
    result = analyze()
    assert "Max Pain Price: N/A (non-unique minimum)" in result
    assert "Max Pain Minimum Interval: $100.00–$120.00" in result
    assert "Max Pain Price: $" not in result


def test_missing_call_oi_does_not_prevent_put_references(options_api):
    ticker, chains, _, analyze = options_api
    ticker.options = ("2026-04-24",)
    calls = _sample_calls()
    del calls["openInterest"]
    chains["2026-04-24"] = _make_options_chain(calls, _sample_puts())
    result = analyze()
    assert "Total Call Open Interest: N/A" in result
    assert "Total Put Open Interest: 83,000" in result
    assert "PCR (Open Interest): N/A" in result
    assert "Max Pain Price: N/A" in result
    assert "| 1 | $180.00 | 45,000 |" in result


def test_next_complete_chain_is_selected_after_incomplete_oi(options_api):
    ticker, chains, _, analyze = options_api
    calls = _sample_calls()
    del calls["openInterest"]
    chains["2026-04-24"] = _make_options_chain(calls, _sample_puts())
    result = analyze()
    assert "Selected Expiration: 2026-05-01" in result
    assert "2026-04-24: incomplete OI" in result
    assert ticker.option_chain.call_count == 2


def test_only_three_candidates_are_queried_and_earliest_nonempty_is_retained(options_api):
    ticker, chains, _, analyze = options_api
    ticker.options = ("2026-04-24", "2026-05-01", "2026-05-08", "2026-05-15")
    calls, puts = _sample_calls(), _sample_puts()
    calls["openInterest"] = puts["openInterest"] = [0, 0, 0]
    for expiry in ticker.options:
        chains[expiry] = _make_options_chain(calls, puts)
    chains["2026-04-24"] = _make_options_chain({}, {})
    chains["2026-05-15"] = _make_options_chain(_sample_calls(), _sample_puts())
    result = analyze()
    assert [call.args[0] for call in ticker.option_chain.call_args_list] == ["2026-04-24", "2026-05-01", "2026-05-08"]
    assert "Selected Expiration: 2026-05-01" in result
    assert "Limited report" in result
    assert "Max Pain Price: $" not in result


def test_all_empty_chains_are_unavailable_without_price_query(options_api):
    ticker, chains, _, analyze = options_api
    for expiry in ticker.options:
        chains[expiry] = _make_options_chain({}, {})
    result = analyze()
    assert "Options chains are empty" in result
    assert "Selected Expiration" not in result
    assert "Max Pain Price: $" not in result


def test_no_future_expirations_are_queried(options_api):
    ticker, _, freeze, analyze = options_api
    freeze("2026-05-02T01:00:00+00:00")
    assert "No future options expiration available" in analyze()
    ticker.option_chain.assert_not_called()


def test_api_failure_on_later_chain_is_not_reported_as_zero_oi(options_api):
    ticker, chains, _, analyze = options_api
    calls, puts = _sample_calls(), _sample_puts()
    calls["openInterest"] = puts["openInterest"] = [0, 0, 0]
    chains["2026-04-24"] = _make_options_chain(calls, puts)

    def fetch(expiry):
        if expiry == "2026-05-01":
            raise ConnectionError("Yahoo unavailable")
        return chains[expiry]

    ticker.option_chain.side_effect = fetch
    result = analyze()
    assert "Error retrieving options analysis" in result
    assert "Yahoo unavailable" in result
    assert "Selected Expiration" not in result
    assert "Max Pain Price" not in result


class TestOptionsRouting:
    def test_options_analysis_in_tools_categories(self):
        assert "get_options_analysis" in TOOLS_CATEGORIES["market_context"]["tools"]

    def test_options_analysis_in_vendor_methods(self):
        assert "yfinance" in VENDOR_METHODS["get_options_analysis"]

    def test_get_category_for_options_analysis(self):
        assert get_category_for_method("get_options_analysis") == "market_context"


class TestMarketAnalystWithOptions:
    def test_market_analyst_has_options_tool_registered(self, mock_context):
        from tradingscope.agents.analysts.market_analyst import create_market_analyst_agent

        mock_context.ticker = "AAPL"
        agent = create_market_analyst_agent(context=mock_context)
        tool_names = [t.name for t in agent.toolkit.tool_groups[0].tools]
        assert "get_options_analysis" in tool_names

    def test_market_analyst_prompt_includes_options(self, mock_context):
        from tradingscope.agents.analysts.market_analyst import create_market_analyst_agent

        mock_context.ticker = "AAPL"
        agent = create_market_analyst_agent(context=mock_context)
        assert "get_options_analysis" in agent._system_prompt
        assert "期权" in agent._system_prompt
        assert "Put/Call Ratio" in agent._system_prompt
