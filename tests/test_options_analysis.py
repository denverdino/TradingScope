"""Unit tests for options analysis functionality."""

import os
import sys
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tradingscope.dataflows.interface import TOOLS_CATEGORIES, VENDOR_METHODS, get_category_for_method
from tradingscope.dataflows.y_finance import get_options_analysis


def _make_options_chain(calls_data, puts_data):
    """Helper to create a mock options chain namedtuple."""
    OptionsChain = namedtuple("OptionsChain", ["calls", "puts"])
    calls_df = pd.DataFrame(calls_data)
    puts_df = pd.DataFrame(puts_data)
    return OptionsChain(calls=calls_df, puts=puts_df)


def _sample_calls():
    return {
        "contractSymbol": ["AAPL260424C00180000", "AAPL260424C00190000", "AAPL260424C00200000"],
        "strike": [180.0, 190.0, 200.0],
        "lastPrice": [22.0, 12.5, 4.0],
        "bid": [21.5, 12.0, 3.8],
        "ask": [22.5, 13.0, 4.2],
        "volume": [1500, 3000, 5000],
        "openInterest": [10000, 25000, 50000],
        "impliedVolatility": [0.35, 0.30, 0.28],
        "inTheMoney": [True, True, False],
    }


def _sample_puts():
    return {
        "contractSymbol": ["AAPL260424P00170000", "AAPL260424P00180000", "AAPL260424P00190000"],
        "strike": [170.0, 180.0, 190.0],
        "lastPrice": [2.0, 5.5, 12.0],
        "bid": [1.8, 5.3, 11.8],
        "ask": [2.2, 5.7, 12.2],
        "volume": [2000, 4000, 1000],
        "openInterest": [30000, 45000, 8000],
        "impliedVolatility": [0.32, 0.29, 0.27],
        "inTheMoney": [False, False, True],
    }


class TestOptionsDataLayer:
    """Test cases for the y_finance.get_options_analysis function."""

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_us_stock_options(self, mock_yf_retry):
        """Test normal options analysis for a US stock."""
        mock_ticker = MagicMock()

        opt_chain = _make_options_chain(_sample_calls(), _sample_puts())

        # yf_retry is called multiple times: options, option_chain, info
        mock_yf_retry.side_effect = [
            ("2026-04-24", "2026-05-01"),  # ticker.options
            opt_chain,  # ticker.option_chain
            {"currentPrice": 192.50},  # ticker.info
        ]

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_options_analysis("AAPL")

        assert "Options Chain Analysis for AAPL" in result
        assert "Put/Call Ratio" in result
        assert "Support Levels" in result
        assert "Resistance Levels" in result
        assert "Max Pain Price" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_no_options_data(self, mock_yf_retry):
        """Test graceful handling when no options data is available."""
        mock_ticker = MagicMock()
        mock_yf_retry.return_value = ()  # empty expirations

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_options_analysis("0700.HK")

        assert "No options data available" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_api_error(self, mock_yf_retry):
        """Test error handling when API call fails."""
        mock_yf_retry.side_effect = Exception("Network error")

        with patch("tradingscope.dataflows.y_finance.yf.Ticker"):
            result = get_options_analysis("AAPL")

        assert "Error retrieving options analysis" in result
        assert "Network error" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_pcr_calculation(self, mock_yf_retry):
        """Test Put/Call Ratio is calculated correctly."""
        mock_ticker = MagicMock()
        opt_chain = _make_options_chain(_sample_calls(), _sample_puts())

        # Total Call OI: 10000 + 25000 + 50000 = 85000
        # Total Put OI: 30000 + 45000 + 8000 = 83000
        # PCR = 83000 / 85000 ≈ 0.98
        mock_yf_retry.side_effect = [
            ("2026-04-24",),
            opt_chain,
            {"currentPrice": 192.50},
        ]

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_options_analysis("AAPL")

        assert "0.98" in result
        assert "中性偏多" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_zero_call_oi(self, mock_yf_retry):
        """Test handling when all call OI is zero (division by zero)."""
        mock_ticker = MagicMock()

        zero_calls = _sample_calls()
        zero_calls["openInterest"] = [0, 0, 0]

        opt_chain = _make_options_chain(zero_calls, _sample_puts())

        mock_yf_retry.side_effect = [
            ("2026-04-24",),
            opt_chain,
            {"currentPrice": 192.50},
        ]

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_options_analysis("AAPL")

        # Should not raise, should show N/A for PCR
        assert "N/A" in result or "Call OI = 0" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_support_resistance_ordering(self, mock_yf_retry):
        """Test that support/resistance levels are sorted by OI descending."""
        mock_ticker = MagicMock()
        opt_chain = _make_options_chain(_sample_calls(), _sample_puts())

        mock_yf_retry.side_effect = [
            ("2026-04-24",),
            opt_chain,
            {"currentPrice": 192.50},
        ]

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_options_analysis("AAPL")

        # Resistance: Call OI is 50000 ($200) > 25000 ($190) > 10000 ($180)
        # So $200 should appear before $190 in resistance section
        resistance_start = result.index("Resistance Levels")
        resistance_section = result[resistance_start:]
        pos_200 = resistance_section.index("200.00")
        pos_190 = resistance_section.index("190.00")
        assert pos_200 < pos_190, "Strike $200 (highest OI) should appear before $190"

        # Support: Put OI is 45000 ($180) > 30000 ($170) > 8000 ($190)
        # So $180 should appear before $170 in support section
        support_start = result.index("Support Levels")
        support_section = result[support_start:resistance_start]
        pos_180 = support_section.index("180.00")
        pos_170 = support_section.index("170.00")
        assert pos_180 < pos_170, "Strike $180 (highest Put OI) should appear before $170"


class TestOptionsRouting:
    """Test cases for options analysis routing configuration."""

    def test_options_analysis_in_tools_categories(self):
        """Verify get_options_analysis is registered in market_context category."""
        assert "get_options_analysis" in TOOLS_CATEGORIES["market_context"]["tools"]

    def test_options_analysis_in_vendor_methods(self):
        """Verify get_options_analysis has a yfinance vendor."""
        assert "get_options_analysis" in VENDOR_METHODS
        assert "yfinance" in VENDOR_METHODS["get_options_analysis"]

    def test_get_category_for_options_analysis(self):
        """Verify routing resolves to market_context category."""
        assert get_category_for_method("get_options_analysis") == "market_context"


class TestMarketAnalystWithOptions:
    """Test cases for Market Analyst agent integration with options tool."""

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"})
    def test_market_analyst_has_options_tool_registered(self):
        """Test that the Market Analyst agent has get_options_analysis registered."""
        from tradingscope.agents.analysts.market_analyst import create_market_analyst_agent
        from tradingscope.agents.utils.context import AgentContext

        context = AgentContext()
        context.company_of_interest = "AAPL"

        agent = create_market_analyst_agent(context=context)
        tool_names = [t.name for t in agent.toolkit.tool_groups[0].tools]
        assert "get_options_analysis" in tool_names

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"})
    def test_market_analyst_prompt_includes_options(self):
        """Test that the system prompt includes options analysis instructions."""
        from tradingscope.agents.analysts.market_analyst import create_market_analyst_agent
        from tradingscope.agents.utils.context import AgentContext

        context = AgentContext()
        context.company_of_interest = "AAPL"

        agent = create_market_analyst_agent(context=context)
        assert "get_options_analysis" in agent._system_prompt
        assert "期权" in agent._system_prompt
        assert "Put/Call Ratio" in agent._system_prompt
