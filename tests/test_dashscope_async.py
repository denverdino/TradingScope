"""Unit tests for async DashScope API calls (evaluator, models, summarize)."""

import asyncio
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentscope.message import UserMsg

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


def _mock_streaming_model(text="Test content"):
    """Build a mock DashScopeChatModel that returns an async generator."""

    async def _mock_call(*args, **kwargs):
        async def _stream():
            chunk = MagicMock()
            chunk.content = [{"type": "text", "text": text}]
            yield chunk

        return _stream()

    mock_model = AsyncMock(side_effect=_mock_call)
    return mock_model


class TestEvaluatorGenerateLesson:
    """Tests for evaluator.py _generate_lesson Agent calls."""

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
        from tradingscope.agents.evaluation.market_outcome import (
            AssessmentStatus,
            TradeAssessment,
        )

        return asyncio.run(
            evaluator._generate_lesson(
                record=record,
                assessment=TradeAssessment(
                    horizon_days=3,
                    status=AssessmentStatus.CORRECT,
                    entry_triggered=True,
                    entry_price=150.0,
                    exit_reason="target",
                    benchmark_return=0.033,
                    strategy_return=0.067,
                    atr_threshold=0.025,
                    limitations=("测试限制",),
                ),
            )
        )

    def test_success(self):
        from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator

        lesson_text = "  [AAPL|2025-01-01|得分:80%] 根因: test 教训: test 改进: test  "
        mock_model = _mock_streaming_model(text=lesson_text)
        middlewares = [object()]
        response = MagicMock()
        response.get_text_content.return_value = lesson_text
        record = self._make_record(entry_price=150.0, target_price=160.0, stop_loss=140.0)
        with patch("tradingscope.agents.evaluation.evaluator.Agent") as mock_agent:
            mock_agent.return_value.reply = AsyncMock(return_value=response)
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = AnalysisEvaluator(
                    model=mock_model,
                    results_dir=tmpdir,
                    dry_run=True,
                    middlewares=middlewares,
                )
                result = self._call_generate_lesson(evaluator, record)

        assert result == lesson_text.strip()
        mock_agent.assert_called_once_with(
            name="EvaluationAgent",
            system_prompt="你是一位客观的投资分析评测专家。",
            model=mock_model,
            middlewares=middlewares,
        )
        mock_agent.return_value.reply.assert_awaited_once()
        message = mock_agent.return_value.reply.await_args.args[0]
        assert message == UserMsg(
            name=message.name,
            content=message.content,
            metadata=message.metadata,
            created_at=message.created_at,
            finished_at=message.finished_at,
            id=message.id,
        )
        assert message.role == "user"
        assert message.name == "evaluator"
        prompt = message.get_text_content()
        assert "AAPL" in prompt
        assert "3个交易日" in prompt
        assert "correct" in prompt
        assert "是" in prompt
        assert "+3.30%" in prompt
        assert "+6.70%" in prompt
        assert "2.50%" in prompt
        assert "target" in prompt
        assert "不得改变上述客观状态" in prompt
        assert "不得把原始分析未提及的信息包装成预测依据" in prompt

    @pytest.mark.parametrize(
        ("status", "required_copy"),
        [
            ("not_filled", "评价执行计划"),
            ("inconclusive", "不得给出确定性的成功或失败结论"),
        ],
    )
    def test_prompt_preserves_non_deterministic_status_semantics(self, status, required_copy):
        from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator
        from tradingscope.agents.evaluation.market_outcome import (
            AssessmentStatus,
            TradeAssessment,
        )

        mock_model = _mock_streaming_model()
        response = MagicMock()
        response.get_text_content.return_value = "评估结果: test\n经验教训: test"
        record = self._make_record()
        assessment = TradeAssessment(
            horizon_days=1,
            status=AssessmentStatus(status),
            entry_triggered=False,
            entry_price=None,
            exit_reason="not_filled" if status == "not_filled" else None,
            benchmark_return=0.01,
            strategy_return=None,
            atr_threshold=0.02,
            limitations=(),
        )

        with patch("tradingscope.agents.evaluation.evaluator.Agent") as mock_agent:
            mock_agent.return_value.reply = AsyncMock(return_value=response)
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = AnalysisEvaluator(model=mock_model, results_dir=tmpdir, dry_run=True)
                asyncio.run(evaluator._generate_lesson(record=record, assessment=assessment))

        prompt = mock_agent.return_value.reply.await_args.args[0].get_text_content()
        assert required_copy in prompt

    def test_each_lesson_uses_a_fresh_agent(self):
        from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator

        mock_model = _mock_streaming_model()
        middlewares = [object()]
        first_agent = MagicMock()
        first_response = MagicMock()
        first_response.get_text_content.return_value = "First lesson"
        first_agent.reply = AsyncMock(return_value=first_response)
        second_agent = MagicMock()
        second_response = MagicMock()
        second_response.get_text_content.return_value = "Second lesson"
        second_agent.reply = AsyncMock(return_value=second_response)
        record = self._make_record()

        with patch(
            "tradingscope.agents.evaluation.evaluator.Agent",
            side_effect=[first_agent, second_agent],
        ) as agent_class:
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = AnalysisEvaluator(
                    model=mock_model,
                    results_dir=tmpdir,
                    dry_run=True,
                    middlewares=middlewares,
                )
                first_result = self._call_generate_lesson(evaluator, record)
                second_result = self._call_generate_lesson(evaluator, record)

        assert first_result == "First lesson"
        assert second_result == "Second lesson"
        assert agent_class.call_count == 2
        for call in agent_class.call_args_list:
            assert call.kwargs["model"] is mock_model
            assert call.kwargs["middlewares"] is middlewares
        first_agent.reply.assert_awaited_once()
        second_agent.reply.assert_awaited_once()

    def test_exception_fallback(self):
        from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator

        mock_model = _mock_streaming_model()
        record = self._make_record()
        with patch("tradingscope.agents.evaluation.evaluator.Agent") as mock_agent:
            mock_agent.return_value.reply = AsyncMock(side_effect=Exception("LLM error"))
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = AnalysisEvaluator(model=mock_model, results_dir=tmpdir, dry_run=True)
                result = self._call_generate_lesson(evaluator, record)

        assert result is None
        mock_agent.return_value.reply.assert_awaited_once()

    def test_empty_response(self):
        from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator

        mock_model = _mock_streaming_model()
        response = MagicMock()
        response.get_text_content.return_value = "   "
        record = self._make_record()
        with patch("tradingscope.agents.evaluation.evaluator.Agent") as mock_agent:
            mock_agent.return_value.reply = AsyncMock(return_value=response)
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = AnalysisEvaluator(model=mock_model, results_dir=tmpdir, dry_run=True)
                result = self._call_generate_lesson(evaluator, record)

        assert result is None
        mock_agent.return_value.reply.assert_awaited_once()
