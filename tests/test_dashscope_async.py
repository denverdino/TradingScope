"""Unit tests for async DashScope summarize calls."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dashscope


def _mock_async_response(status_code=200, content="Test content"):
    """Build a mock DashScope async Generation response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.code = ""
    mock.message = ""
    choice = MagicMock()
    choice.message.content = content
    mock.output.choices = [choice]
    return mock


class TestSummarizeForMemory:
    """Tests for summarize.py async DashScope calls."""

    def test_success(self):
        from tradingscope.agents.utils.summarize import summarize_for_memory

        with patch.object(dashscope.AioGeneration, "call", new=AsyncMock(return_value=_mock_async_response(content="Short summary"))) as mock_call:
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
                long_content = "A" * 600
                result = asyncio.run(summarize_for_memory(long_content, max_chars=500))
                assert "Short summary" in result

    def test_missing_api_key_fallback(self):
        from tradingscope.agents.utils.summarize import summarize_for_memory

        with patch.dict(os.environ, {}, clear=True):
            long_content = "A" * 100
            result = asyncio.run(summarize_for_memory(long_content, max_chars=50))
            assert len(result) <= 50

    def test_api_error_fallback(self):
        from tradingscope.agents.utils.summarize import summarize_for_memory

        with patch.object(dashscope.AioGeneration, "call", new=AsyncMock(return_value=_mock_async_response(status_code=500))) as mock_call:
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
                long_content = "A" * 100
                result = asyncio.run(summarize_for_memory(long_content, max_chars=50))
                assert len(result) <= 50

    def test_exception_fallback(self):
        from tradingscope.agents.utils.summarize import summarize_for_memory

        async_mock = AsyncMock(side_effect=Exception("SDK error"))
        with patch.object(dashscope.AioGeneration, "call", async_mock):
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
                long_content = "A" * 100
                result = asyncio.run(summarize_for_memory(long_content, max_chars=50))
                assert len(result) <= 50

    def test_short_content_no_api_call(self):
        from tradingscope.agents.utils.summarize import summarize_for_memory

        result = asyncio.run(summarize_for_memory("Short", max_chars=500))
        assert result == "Short"
