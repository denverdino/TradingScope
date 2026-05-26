"""Unit tests for volume analysis functionality."""

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tradingscope.dataflows.interface import TOOLS_CATEGORIES, VENDOR_METHODS, get_category_for_method
from tradingscope.dataflows.y_finance import get_volume_analysis


def _make_history(days=40, base_price=100.0, base_volume=1_000_000, trend="up"):
    """Helper to create mock historical data DataFrame."""
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=days)
    n = len(dates)
    prices = []
    volumes = []
    price = base_price

    for i in range(n):
        if trend == "up":
            change = 0.5 + (i * 0.1)
            vol_mult = 1.0 + (i * 0.02)
        elif trend == "down":
            change = -(0.5 + (i * 0.1))
            vol_mult = 1.0 + (i * 0.02)
        elif trend == "divergence_bearish":
            # Price up but volume declining sharply
            change = 0.5 + (i * 0.1)
            vol_mult = max(0.1, 1.5 - (i * 0.05))
        elif trend == "divergence_bullish":
            # Price down but volume increasing
            change = -(0.5 + (i * 0.1))
            vol_mult = 1.0 + (i * 0.03)
        else:
            change = 0.3 * ((-1) ** i)
            vol_mult = 1.0

        price = price + change
        prices.append(price)
        volumes.append(int(base_volume * vol_mult))

    data = pd.DataFrame(
        {
            "Open": [p - 0.5 for p in prices],
            "High": [p + 1.0 for p in prices],
            "Low": [p - 1.0 for p in prices],
            "Close": prices,
            "Volume": volumes,
        },
        index=dates,
    )
    return data


class TestVolumeAnalysisDataLayer:
    """Test cases for the y_finance.get_volume_analysis function."""

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_normal_volume_analysis(self, mock_yf_retry):
        """Test normal volume analysis returns all expected sections."""
        mock_ticker = MagicMock()
        hist = _make_history(days=40, trend="up")
        mock_yf_retry.return_value = hist

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_volume_analysis("AAPL")

        assert "Volume Analysis for AAPL" in result
        assert "Daily Volume Metrics" in result
        assert "Volume Moving Average Comparison" in result
        assert "Volume Expansion/Contraction Phase" in result
        assert "OBV Trend Analysis" in result
        assert "Volume-Price Divergence" in result
        assert "Volume Distribution (Up/Down Days)" in result
        assert "Volume Analysis Summary" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_empty_data(self, mock_yf_retry):
        """Test graceful handling when no data is available."""
        mock_ticker = MagicMock()
        mock_yf_retry.return_value = pd.DataFrame()

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_volume_analysis("INVALID")

        assert "No historical data found" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_api_error(self, mock_yf_retry):
        """Test error handling when API call fails."""
        mock_yf_retry.side_effect = Exception("Network error")

        with patch("tradingscope.dataflows.y_finance.yf.Ticker"):
            result = get_volume_analysis("AAPL")

        assert "Error retrieving volume analysis" in result
        assert "Network error" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_volume_expansion(self, mock_yf_retry):
        """Test volume expansion detection with high volume data."""
        mock_ticker = MagicMock()
        # Create data where latest volume is much higher than average
        hist = _make_history(days=40, base_volume=1_000_000, trend="up")
        # Boost last day's volume to 2x
        hist.iloc[-1, hist.columns.get_loc("Volume")] = 3_000_000
        mock_yf_retry.return_value = hist

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_volume_analysis("AAPL")

        assert "放量" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_volume_contraction(self, mock_yf_retry):
        """Test volume contraction detection with low volume data."""
        mock_ticker = MagicMock()
        hist = _make_history(days=40, base_volume=1_000_000, trend="flat")
        # Set last day's volume very low
        hist.iloc[-1, hist.columns.get_loc("Volume")] = 200_000
        mock_yf_retry.return_value = hist

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_volume_analysis("AAPL")

        assert "缩量" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_obv_uptrend_confirmation(self, mock_yf_retry):
        """Test OBV confirms uptrend when price and volume both rise."""
        mock_ticker = MagicMock()
        hist = _make_history(days=40, trend="up")
        mock_yf_retry.return_value = hist

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_volume_analysis("AAPL")

        assert "OBV" in result
        # With consistently rising prices, OBV should be rising
        assert "上升" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_bearish_divergence(self, mock_yf_retry):
        """Test bearish divergence: price rising but volume declining."""
        mock_ticker = MagicMock()
        hist = _make_history(days=40, trend="divergence_bearish")
        mock_yf_retry.return_value = hist

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_volume_analysis("AAPL")

        assert "看跌背离" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_bullish_divergence(self, mock_yf_retry):
        """Test bullish divergence: price falling but volume increasing."""
        mock_ticker = MagicMock()
        hist = _make_history(days=40, trend="divergence_bullish")
        mock_yf_retry.return_value = hist

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_volume_analysis("AAPL")

        assert "看涨背离" in result

    @patch("tradingscope.dataflows.y_finance.yf_retry")
    def test_up_down_distribution(self, mock_yf_retry):
        """Test volume distribution by up/down days."""
        mock_ticker = MagicMock()
        hist = _make_history(days=40, trend="up")
        mock_yf_retry.return_value = hist

        with patch("tradingscope.dataflows.y_finance.yf.Ticker", return_value=mock_ticker):
            result = get_volume_analysis("AAPL")

        assert "Up Days" in result
        assert "Down Days" in result
        assert "Up/Down Volume Ratio" in result


class TestVolumeAnalysisRouting:
    """Test cases for volume analysis routing configuration."""

    def test_volume_analysis_in_tools_categories(self):
        """Verify get_volume_analysis is registered in market_context category."""
        assert "get_volume_analysis" in TOOLS_CATEGORIES["market_context"]["tools"]

    def test_volume_analysis_in_vendor_methods(self):
        """Verify get_volume_analysis has a yfinance vendor."""
        assert "get_volume_analysis" in VENDOR_METHODS
        assert "yfinance" in VENDOR_METHODS["get_volume_analysis"]

    def test_get_category_for_volume_analysis(self):
        """Verify routing resolves to market_context category."""
        assert get_category_for_method("get_volume_analysis") == "market_context"


class TestMarketAnalystWithVolumeAnalysis:
    """Test cases for Market Analyst agent integration with volume analysis tool."""

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"})
    def test_market_analyst_has_volume_tool_registered(self):
        """Test that the Market Analyst agent has get_volume_analysis registered."""
        from tradingscope.agents.analysts.market_analyst import create_market_analyst_agent
        from tradingscope.agents.utils.context import AgentContext

        context = AgentContext()
        context.company_of_interest = "AAPL"

        agent = create_market_analyst_agent(context=context)
        tool_names = [t.name for t in agent.toolkit.tool_groups[0].tools]
        assert "get_volume_analysis" in tool_names

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"})
    def test_market_analyst_prompt_includes_volume(self):
        """Test that the system prompt includes volume analysis instructions."""
        from tradingscope.agents.analysts.market_analyst import create_market_analyst_agent
        from tradingscope.agents.utils.context import AgentContext

        context = AgentContext()
        context.company_of_interest = "AAPL"

        agent = create_market_analyst_agent(context=context)
        assert "get_volume_analysis" in agent._system_prompt
        assert "成交量" in agent._system_prompt
        assert "OBV" in agent._system_prompt
