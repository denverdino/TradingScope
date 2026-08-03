"""Post-market analysis evaluator.

Scores previous analysis records against actual market data and
generates Lessons Learned for the shared memory namespace.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, List, Optional

from agentscope import logger
from agentscope.agent import Agent
from agentscope.message import UserMsg
from agentscope.model import DashScopeChatModel

from .market_outcome import (
    AssessmentStatus,
    TradeAssessment,
    assess_trade,
    parse_market_bars,
)
from .models import AnalysisRecord, EvaluationResult
from .oss_store import OSSAnalysisStore


def calculate_lesson_weight(
    lesson_type: str,
    error_weight_multiplier: float = 2.0,
) -> float:
    """Calculate memory weight for a lesson.

    Error cases get higher weight to counter self-reinforcing bias.

    Args:
        lesson_type: success/failure/partial
        error_weight_multiplier: multiplier for error cases

    Returns:
        Weight between 0 and 1
    """
    if lesson_type == "failure":
        base_weight = 0.8 * error_weight_multiplier
    elif lesson_type == "partial":
        base_weight = 0.6 * error_weight_multiplier
    else:  # success
        base_weight = 0.4

    return min(1.0, max(0.1, base_weight))


_LESSON_PROMPT = """你是一位客观的投资分析评测专家。请解释已经由程序计算出的交易评估结果，并生成经验教训。

## 分析记录
- 股票: {ticker}
- 分析日期: {trade_date}
- 预测方向: {direction}
- 操作建议: {action}
- 置信度: {confidence:.0%}
- 入场价: {entry_price}
- 目标价: {target_price}
- 止损价: {stop_loss}
- 核心理由: {reasoning}

## 原始分析报告（摘要）
{report_excerpt}

## 客观评估结果
- 评估周期: {horizon_days}个交易日
- 客观状态: {status}
- 是否成交: {entry_triggered}
- 实际成交价: {actual_entry_price}
- 基准收益率: {benchmark_return}
- 策略收益率: {strategy_return}
- ATR有效波动边界: {atr_threshold}
- 退出原因: {exit_reason}
- 评估限制: {limitations}

## 输出要求
请用中文输出，严格控制在450字符以内。格式如下：
[{ticker}|{trade_date}|{horizon_days}日]
评估结果: (解释客观评估状态及交易计划的执行情况，1-2句)
经验教训: (核心经验教训，2-3句)

不得改变上述客观状态。不得把原始分析未提及的信息包装成预测依据。
必须区分方向判断与交易计划可执行性。状态为not_filled时只评价执行计划，不得评价为方向错误。
状态为inconclusive时不得给出确定性的成功或失败结论。
不要使用Markdown格式，直接输出纯文本。"""


class AnalysisEvaluator:
    """Evaluates analysis records against actual post-market data.

    Fetches real stock prices, scores the analysis accuracy, and uses
    an LLM to generate structured Lessons Learned that are stored in
    the shared lessons_learned memory namespace.
    """

    def __init__(
        self,
        model: DashScopeChatModel,
        memory_manager: Optional[object] = None,
        results_dir: Optional[str] = None,
        dry_run: bool = False,
        middlewares: list[Any] | None = None,
    ):
        """Initialize AnalysisEvaluator.

        Args:
            model: DashScopeChatModel instance for LLM calls.
            memory_manager: FinancialMemoryManager for writing lessons.
                            If None, lessons are generated but not stored.
            results_dir: Directory for local tracking files.
            dry_run: If True, skip all side effects (memory writes, record marking).
            middlewares: AgentScope middleware applied to evaluation calls.
        """
        self._model = model
        self._middlewares = middlewares
        self._memory_manager = memory_manager
        self._record_store = OSSAnalysisStore(results_dir=results_dir)
        self._dry_run = dry_run
        self._get_stock_data = None

    def _ensure_data_imports(self) -> bool:
        """Lazily import data fetching functions."""
        if self._get_stock_data is not None:
            return True
        try:
            from tradingscope.dataflows.interface import route_to_vendor

            self._get_stock_data = lambda symbol, start, end: route_to_vendor("get_stock_data", symbol, start, end)
            return True
        except ImportError as e:
            logger.warning("Failed to import data functions: %s", e)
            return False

    async def run_batch_evaluation(
        self,
        tickers: Optional[List[str]] = None,
        date: Optional[str] = None,
    ) -> List[EvaluationResult]:
        """Evaluate all pending analysis records from OSS.

        Args:
            tickers: List of stock symbols to evaluate (optional)
            date: Filter by trade date (optional)

        Returns:
            List of EvaluationResult with ticker, evaluation and lesson fields
        """
        today = datetime.now().strftime("%Y-%m-%d")
        pending = await self._record_store.list_pending(before_date=today, tickers=tickers, date=date)

        logger.info("[Evaluator] Found %d pending records", len(pending))

        results: List[EvaluationResult] = []
        for record in pending:
            try:
                record_results = await self.evaluate_single(record)
                results.extend(record_results)
            except Exception as e:
                logger.warning(
                    "[Evaluator] Error evaluating %s/%s: %s",
                    record.ticker,
                    record.trade_date,
                    e,
                )

        logger.info("[Evaluator] Generated %d results", len(results))
        return results

    async def evaluate_single(self, record: AnalysisRecord) -> List[EvaluationResult]:
        """Evaluate a single analysis record.

        Args:
            record: AnalysisRecord to evaluate

        Returns:
            One result per mature, pending horizon; empty when data is unavailable.
        """
        if not self._ensure_data_imports():
            return []

        trade_date = datetime.strptime(record.trade_date, "%Y-%m-%d")
        start_date = (trade_date - timedelta(days=35)).strftime("%Y-%m-%d")
        end_date = (trade_date + timedelta(days=14)).strftime("%Y-%m-%d")

        stock_data = self._get_stock_data(record.ticker, start_date, end_date)
        if not stock_data or "Error" in str(stock_data):
            logger.warning("[Evaluator] No stock data for %s", record.ticker)
            return []

        try:
            bars = parse_market_bars(_strip_vendor_preamble(stock_data))
        except ValueError as exc:
            logger.warning("[Evaluator] Invalid stock data for %s: %s", record.ticker, exc)
            return []

        analysis_date = trade_date.date()
        history_count = sum(bar.trade_date < analysis_date for bar in bars)
        evaluation_session_count = sum(bar.trade_date >= analysis_date for bar in bars)

        results: List[EvaluationResult] = []
        for horizon_days in (1, 3, 5):
            if self._record_store.is_evaluated(record.ticker, record.trade_date, horizon_days):
                continue
            try:
                assessment = assess_trade(record, bars, horizon_days)
                if assessment is None:
                    logger.info(
                        "[Evaluator] Skipping %s/%s/%s: insufficient market data "
                        "(history=%s, required_history=15, evaluation_sessions=%s, required_sessions=%s)",
                        record.ticker,
                        record.trade_date,
                        horizon_days,
                        history_count,
                        evaluation_session_count,
                        horizon_days,
                    )
                    continue
                lesson_content = await self._generate_lesson(record, assessment)
                if not lesson_content:
                    continue

                lesson_type = _lesson_type_for_status(assessment.status)
                weight = calculate_lesson_weight(lesson_type=lesson_type)
                if self._memory_manager and not self._dry_run:
                    lessons_mem = self._memory_manager.lessons_memory
                    if lessons_mem:
                        await lessons_mem.add_reflection_lesson(
                            lesson_content=lesson_content,
                            weight=weight,
                            lesson_type=lesson_type,
                        )
                        logger.info(
                            "[Evaluator] Stored lesson for %s/%s/%s (type=%s, weight=%.2f)",
                            record.ticker,
                            record.trade_date,
                            horizon_days,
                            lesson_type,
                            weight,
                        )

                if not self._dry_run:
                    self._record_store.mark_evaluated(record.ticker, record.trade_date, horizon_days)

                evaluation, lesson = _parse_evaluation_and_lesson(lesson_content)
                results.append(
                    EvaluationResult(
                        ticker=record.ticker,
                        evaluation=evaluation,
                        lesson=lesson,
                        trade_date=record.trade_date,
                        horizon_days=horizon_days,
                        status=assessment.status.value,
                        entry_triggered=assessment.entry_triggered,
                        benchmark_return=assessment.benchmark_return,
                        strategy_return=assessment.strategy_return,
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "[Evaluator] Error evaluating %s/%s/%s: %s",
                    record.ticker,
                    record.trade_date,
                    horizon_days,
                    exc,
                )
        return results

    async def _generate_lesson(
        self,
        record: AnalysisRecord,
        assessment: TradeAssessment,
    ) -> Optional[str]:
        """Generate a structured lesson using LLM."""
        logger.info("[Evaluator] record: %s", record)

        prompt = _LESSON_PROMPT.format(
            ticker=record.ticker,
            trade_date=record.trade_date,
            direction=record.direction,
            action=record.action,
            confidence=record.confidence,
            entry_price=record.entry_price or "-",
            target_price=record.target_price or "-",
            stop_loss=record.stop_loss or "-",
            reasoning=record.reasoning or "-",
            report_excerpt=record.final_decision_summary or "-",
            horizon_days=assessment.horizon_days,
            status=assessment.status.value,
            entry_triggered="是" if assessment.entry_triggered else "否",
            actual_entry_price=_format_price(assessment.entry_price),
            benchmark_return=f"{assessment.benchmark_return:+.2%}",
            strategy_return=_format_percent(assessment.strategy_return),
            atr_threshold=f"{assessment.atr_threshold:.2%}",
            exit_reason=assessment.exit_reason or "-",
            limitations="；".join(assessment.limitations) or "无",
        )

        logger.info("[Evaluator] Prompt: %s", prompt)

        try:
            agent = Agent(
                name="EvaluationAgent",
                system_prompt="你是一位客观的投资分析评测专家。",
                model=self._model,
                middlewares=self._middlewares,
            )
            response = await agent.reply(
                UserMsg(name="evaluator", content=prompt),
            )
            text = response.get_text_content().strip()
            return text or None
        except Exception as e:
            logger.warning("[Evaluator] LLM call failed: %s", e)
            return None


def _lesson_type_for_status(status: AssessmentStatus) -> str:
    if status is AssessmentStatus.CORRECT:
        return "success"
    if status is AssessmentStatus.INCORRECT:
        return "failure"
    return "partial"


def _format_price(value: float | None) -> str:
    return f"${value:.2f}" if value is not None else "-"


def _format_percent(value: float | None) -> str:
    return f"{value:+.2%}" if value is not None else "-"


def _strip_vendor_preamble(csv_text: str) -> str:
    """Remove yfinance metadata lines before strict CSV parsing."""
    return "\n".join(line for line in csv_text.splitlines() if line.strip() and not line.lstrip().startswith("#"))


def _parse_evaluation_and_lesson(lesson_content: str) -> tuple[str, str]:
    """Parse LLM lesson text into (evaluation, lesson) parts.

    Expects format:
        [TICKER|DATE]
        评估: ...
        教训: ...

    Returns a tuple of (evaluation, lesson). Falls back to (lesson_content, "")
    if the expected format is not found.
    """
    import re

    evaluation = ""
    lesson = ""

    eval_match = re.search(r"评估结果[:：]\s*(.+?)(?=经验教训[:：]|$)", lesson_content, re.DOTALL)
    lesson_match = re.search(r"经验教训[:：]\s*(.+)", lesson_content, re.DOTALL)

    if eval_match:
        evaluation = eval_match.group(1).strip()
    if lesson_match:
        lesson = lesson_match.group(1).strip()

    if not evaluation and not lesson:
        evaluation = lesson_content.strip()

    return evaluation, lesson
