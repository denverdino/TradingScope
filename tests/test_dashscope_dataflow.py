"""Unit tests for DashScope dataflow functions."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _mock_response(status_code=200, content="Test content", code=None, message=None):
    """Build a mock DashScope Generation response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.code = code or ""
    mock.message = message or ""
    choice = MagicMock()
    choice.message.content = content
    mock.output.choices = [choice]
    return mock


def _mock_malformed_response():
    """Build a mock response with None output (malformed)."""
    mock = MagicMock()
    mock.status_code = 200
    mock.output = None
    return mock


def _mock_empty_choices_response():
    """Build a mock response with empty choices list."""
    mock = MagicMock()
    mock.status_code = 200
    mock.output.choices = []
    return mock


class TestExtractDashscopeContent:
    """Tests for _extract_dashscope_content helper."""

    def test_normal_response(self):
        from tradingscope.dataflows.dashscope import _extract_dashscope_content

        response = _mock_response(content="News about AAPL")
        assert _extract_dashscope_content(response) == "News about AAPL"

    def test_none_output(self):
        from tradingscope.dataflows.dashscope import _extract_dashscope_content

        response = _mock_malformed_response()
        assert _extract_dashscope_content(response) == ""

    def test_empty_choices(self):
        from tradingscope.dataflows.dashscope import _extract_dashscope_content

        response = _mock_empty_choices_response()
        assert _extract_dashscope_content(response) == ""

    def test_none_message(self):
        from tradingscope.dataflows.dashscope import _extract_dashscope_content

        response = MagicMock()
        response.status_code = 200
        choice = MagicMock()
        choice.message = None
        response.output.choices = [choice]
        assert _extract_dashscope_content(response) == ""

    def test_empty_content(self):
        from tradingscope.dataflows.dashscope import _extract_dashscope_content

        response = MagicMock()
        response.status_code = 200
        choice = MagicMock()
        choice.message.content = ""
        response.output.choices = [choice]
        assert _extract_dashscope_content(response) == ""

    def test_none_content(self):
        from tradingscope.dataflows.dashscope import _extract_dashscope_content

        response = MagicMock()
        response.status_code = 200
        choice = MagicMock()
        choice.message.content = None
        response.output.choices = [choice]
        assert _extract_dashscope_content(response) == ""


class TestGetStockNewsDashscope:
    """Tests for get_stock_news_dashscope."""

    @patch("tradingscope.dataflows.dashscope.dashscope.Generation.call")
    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_success(self, mock_config, mock_call):
        from tradingscope.dataflows.dashscope import get_stock_news_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        mock_call.return_value = _mock_response(content="News about AAPL")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            result = get_stock_news_dashscope("AAPL", "2025-01-01", "2025-01-31")
        assert result == "News about AAPL"

    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_missing_api_key(self, mock_config):
        from tradingscope.dataflows.dashscope import get_stock_news_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
                get_stock_news_dashscope("AAPL", "2025-01-01", "2025-01-31")

    @patch("tradingscope.dataflows.dashscope.dashscope.Generation.call")
    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_api_error(self, mock_config, mock_call):
        from tradingscope.dataflows.dashscope import get_stock_news_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        mock_call.return_value = _mock_response(status_code=400, message="Invalid request")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            with pytest.raises(RuntimeError, match="DashScope API error"):
                get_stock_news_dashscope("AAPL", "2025-01-01", "2025-01-31")

    @patch("tradingscope.dataflows.dashscope.dashscope.Generation.call")
    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_empty_content(self, mock_config, mock_call):
        from tradingscope.dataflows.dashscope import get_stock_news_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        mock_call.return_value = _mock_response(content="")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            result = get_stock_news_dashscope("AAPL", "2025-01-01", "2025-01-31")
        assert result == ""

    @patch("tradingscope.dataflows.dashscope.dashscope.Generation.call")
    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_sdk_exception(self, mock_config, mock_call):
        from dashscope.common.error import DashScopeException

        from tradingscope.dataflows.dashscope import get_stock_news_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        mock_call.side_effect = DashScopeException("DashScope SDK error")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            with pytest.raises(RuntimeError, match="DashScope call failed"):
                get_stock_news_dashscope("AAPL", "2025-01-01", "2025-01-31")

    @patch("tradingscope.dataflows.dashscope.dashscope.Generation.call")
    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_connection_error(self, mock_config, mock_call):
        from tradingscope.dataflows.dashscope import get_stock_news_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        mock_call.side_effect = ConnectionError("Network down")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            with pytest.raises(RuntimeError, match="DashScope call failed"):
                get_stock_news_dashscope("AAPL", "2025-01-01", "2025-01-31")

    @patch("tradingscope.dataflows.dashscope.dashscope.Generation.call")
    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_malformed_response(self, mock_config, mock_call):
        from tradingscope.dataflows.dashscope import get_stock_news_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        mock_call.return_value = _mock_malformed_response()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            result = get_stock_news_dashscope("AAPL", "2025-01-01", "2025-01-31")
        assert result == ""


class TestGetGlobalNewsDashscope:
    """Tests for get_global_news_dashscope."""

    @patch("tradingscope.dataflows.dashscope.dashscope.Generation.call")
    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_success(self, mock_config, mock_call):
        from tradingscope.dataflows.dashscope import get_global_news_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        mock_call.return_value = _mock_response(content="Global macro news")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            result = get_global_news_dashscope("2025-01-31")
        assert result == "Global macro news"

    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_missing_api_key(self, mock_config):
        from tradingscope.dataflows.dashscope import get_global_news_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
                get_global_news_dashscope("2025-01-31")

    @patch("tradingscope.dataflows.dashscope.dashscope.Generation.call")
    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_api_error(self, mock_config, mock_call):
        from tradingscope.dataflows.dashscope import get_global_news_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        mock_call.return_value = _mock_response(status_code=500, message="Server error")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            with pytest.raises(RuntimeError, match="DashScope API error"):
                get_global_news_dashscope("2025-01-31")


class TestGetFundamentalsDashscope:
    """Tests for get_fundamentals_dashscope."""

    @patch("tradingscope.dataflows.dashscope.dashscope.Generation.call")
    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_success(self, mock_config, mock_call):
        from tradingscope.dataflows.dashscope import get_fundamentals_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        mock_call.return_value = _mock_response(content="PE ratio: 25.3, PS: 8.1")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            result = get_fundamentals_dashscope("AAPL", "2025-01-31")
        assert "PE" in result

    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_missing_api_key(self, mock_config):
        from tradingscope.dataflows.dashscope import get_fundamentals_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
                get_fundamentals_dashscope("AAPL", "2025-01-31")

    @patch("tradingscope.dataflows.dashscope.dashscope.Generation.call")
    @patch("tradingscope.dataflows.dashscope.get_config")
    def test_api_error(self, mock_config, mock_call):
        from tradingscope.dataflows.dashscope import get_fundamentals_dashscope

        mock_config.return_value = {"quick_think_llm": "qwen3.5-flash"}
        mock_call.return_value = _mock_response(status_code=403, message="Forbidden")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            with pytest.raises(RuntimeError, match="DashScope API error"):
                get_fundamentals_dashscope("AAPL", "2025-01-31")
