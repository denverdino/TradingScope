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


class TestSummarizeDecision:
    """Tests for models.py _summarize_decision async DashScope calls."""

    def test_success(self):
        from tradingscope.agents.evaluation.models import _summarize_decision

        with patch.object(dashscope.AioGeneration, "call", new=AsyncMock(return_value=_mock_async_response(content="Key factors summary"))) as mock_call:
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
                long_content = "A" * 600
                result = asyncio.run(_summarize_decision(long_content))
                assert "Key factors" in result

    def test_missing_api_key_fallback(self):
        from tradingscope.agents.evaluation.models import _summarize_decision

        with patch.dict(os.environ, {}, clear=True):
            long_content = "A" * 600
            result = asyncio.run(_summarize_decision(long_content))
            assert len(result) <= 500

    def test_api_error_fallback(self):
        from tradingscope.agents.evaluation.models import _summarize_decision

        with patch.object(dashscope.AioGeneration, "call", new=AsyncMock(return_value=_mock_async_response(status_code=400))) as mock_call:
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
                long_content = "A" * 600
                result = asyncio.run(_summarize_decision(long_content))
                assert len(result) <= 500

    def test_exception_fallback(self):
        from tradingscope.agents.evaluation.models import _summarize_decision

        async_mock = AsyncMock(side_effect=Exception("Connection failed"))
        with patch.object(dashscope.AioGeneration, "call", async_mock):
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
                long_content = "A" * 600
                result = asyncio.run(_summarize_decision(long_content))
                assert len(result) <= 500

    def test_empty_content(self):
        from tradingscope.agents.evaluation.models import _summarize_decision

        result = asyncio.run(_summarize_decision(""))
        assert result == ""

    def test_short_content_no_api_call(self):
        from tradingscope.agents.evaluation.models import _summarize_decision

        result = asyncio.run(_summarize_decision("Short decision"))
        assert result == "Short decision"


class TestEvaluatorGenerateLesson:
    """Tests for evaluator.py _generate_lesson async DashScope calls."""

    def test_success(self):
        from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator
        from tradingscope.agents.evaluation.models import AnalysisRecord

        with patch.object(dashscope.AioGeneration, "call", new=AsyncMock(return_value=_mock_async_response(content="[AAPL|2025-01-01|得分:80%] 根因: test 教训: test 改进: test"))) as mock_call:
            record = AnalysisRecord(
                ticker="AAPL",
                trade_date="2025-01-01",
                direction="bullish",
                action="buy",
                confidence=0.8,
                entry_price=150.0,
                target_price=160.0,
                stop_loss=140.0,
                reasoning="Strong fundamentals",
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = AnalysisEvaluator(results_dir=tmpdir, dry_run=True)
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
                    result = asyncio.run(
                        evaluator._generate_lesson(
                            record=record,
                            eval_date="2025-01-02",
                            price_t=150.0,
                            price_tn=155.0,
                            actual_return=0.033,
                            direction_correct=True,
                            target_reached=False,
                            stop_loss_triggered=False,
                            accuracy_score=0.8,
                        )
                    )
                    assert result is not None
                    assert "AAPL" in result

    def test_missing_api_key_fallback(self):
        from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator
        from tradingscope.agents.evaluation.models import AnalysisRecord

        record = AnalysisRecord(
            ticker="AAPL",
            trade_date="2025-01-01",
            direction="bullish",
            action="buy",
            confidence=0.8,
            reasoning="Strong fundamentals",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = AnalysisEvaluator(results_dir=tmpdir, dry_run=True)
            with patch.dict(os.environ, {}, clear=True):
                result = asyncio.run(
                    evaluator._generate_lesson(
                        record=record,
                        eval_date="2025-01-02",
                        price_t=150.0,
                        price_tn=155.0,
                        actual_return=0.033,
                        direction_correct=True,
                        target_reached=False,
                        stop_loss_triggered=False,
                        accuracy_score=0.8,
                    )
                )
                assert result is not None
                assert "AAPL" in result

    def test_api_error_fallback(self):
        from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator
        from tradingscope.agents.evaluation.models import AnalysisRecord

        with patch.object(dashscope.AioGeneration, "call", new=AsyncMock(return_value=_mock_async_response(status_code=500))) as mock_call:
            record = AnalysisRecord(
                ticker="AAPL",
                trade_date="2025-01-01",
                direction="bullish",
                action="buy",
                confidence=0.8,
                reasoning="Strong fundamentals",
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = AnalysisEvaluator(results_dir=tmpdir, dry_run=True)
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
                    result = asyncio.run(
                        evaluator._generate_lesson(
                            record=record,
                            eval_date="2025-01-02",
                            price_t=150.0,
                            price_tn=155.0,
                            actual_return=0.033,
                            direction_correct=True,
                            target_reached=False,
                            stop_loss_triggered=False,
                            accuracy_score=0.8,
                        )
                    )
                    assert result is not None
