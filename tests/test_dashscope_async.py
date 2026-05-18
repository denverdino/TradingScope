"""Unit tests for async DashScope API calls (evaluator, models, summarize)."""

import asyncio
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def _mock_multimodal_response(status_code=200, text="Test content"):
    """Build a mock DashScope MultiModalConversation response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.code = ""
    mock.message = ""
    choice = MagicMock()
    choice.message.content = [{"text": text}]
    mock.output.choices = [choice]
    return mock


class TestEvaluatorGenerateLesson:
    """Tests for evaluator.py _generate_lesson DashScope calls."""

    def _make_record(self, **kwargs):
        from tradingscope.agents.evaluation.models import AnalysisRecord

        defaults = {
            "ticker": "AAPL",
            "trade_date": "2025-01-01",
            "direction": "bullish",
            "action": "buy",
            "confidence": 0.8,
            "reasoning": "Strong fundamentals",
        }
        return AnalysisRecord(**{**defaults, **kwargs})

    def _call_generate_lesson(self, evaluator, record):
        return asyncio.run(
            evaluator._generate_lesson(
                record=record,
                price_prev=150.0,
                price_t=155.0,
                actual_return=0.033,
                direction_correct=True,
                stop_loss_triggered=False,
            )
        )

    def test_success(self):
        from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator

        lesson_text = "[AAPL|2025-01-01|得分:80%] 根因: test 教训: test 改进: test"
        mock_resp = _mock_multimodal_response(text=lesson_text)
        with patch.object(dashscope.MultiModalConversation, "call", return_value=mock_resp):
            record = self._make_record(entry_price=150.0, target_price=160.0, stop_loss=140.0)
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = AnalysisEvaluator(results_dir=tmpdir, dry_run=True)
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
                    result = self._call_generate_lesson(evaluator, record)
                    assert result is not None
                    assert "AAPL" in result

    def test_missing_api_key_fallback(self):
        from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator

        record = self._make_record()
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = AnalysisEvaluator(results_dir=tmpdir, dry_run=True)
            with patch.dict(os.environ, {}, clear=True):
                result = self._call_generate_lesson(evaluator, record)
                assert result is None

    def test_api_error_fallback(self):
        from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator

        mock_resp = _mock_multimodal_response(status_code=500)
        with patch.object(dashscope.MultiModalConversation, "call", return_value=mock_resp):
            record = self._make_record()
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = AnalysisEvaluator(results_dir=tmpdir, dry_run=True)
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
                    result = self._call_generate_lesson(evaluator, record)
                    assert result is None
