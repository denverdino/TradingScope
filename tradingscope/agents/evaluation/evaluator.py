"""Post-market analysis evaluator.

Scores previous analysis records against actual market data and
generates Lessons Learned for the shared memory namespace.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Optional

import dashscope
from agentscope import logger

from tradingscope.default_config import DEFAULT_CONFIG

from .models import AnalysisRecord
from .oss_store import OSSAnalysisStore


def calculate_lesson_weight(
    lesson_type: str,
    accuracy_score: float,
    confidence: float,
    error_weight_multiplier: float = 2.0,
) -> float:
    """Calculate memory weight for a lesson.

    Error cases get higher weight to counter self-reinforcing bias.

    Args:
        lesson_type: success/failure/partial
        accuracy_score: 0-1 prediction accuracy
        confidence: 0-1 original prediction confidence
        error_weight_multiplier: multiplier for error cases

    Returns:
        Weight between 0 and 1
    """
    base_weight = 0.5

    if lesson_type == "failure":
        base_weight = 0.8 * error_weight_multiplier
    elif lesson_type == "partial":
        base_weight = 0.6 * error_weight_multiplier
    else:  # success
        base_weight = 0.4

    # High confidence + low accuracy = overconfidence penalty
    if confidence > 0.7 and accuracy_score < 0.4:
        base_weight *= 1.2

    # Low confidence + high accuracy = missed opportunity
    if confidence < 0.4 and accuracy_score > 0.7:
        base_weight *= 1.1

    return min(1.0, max(0.1, base_weight))


_LESSON_PROMPT = """你是一位客观的投资分析评测专家。请根据以下信息评估前一天的交易分析记录，并生成经验教训。

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

## 实际市场数据
- 评估日期: {eval_date}
- 分析日收盘价: {price_t}
- 最新收盘价: {price_tn}
- 实际收益率: {actual_return}
- 方向判断: {direction_result}
- 目标达成: {target_result}
- 止损触发: {stop_loss_result}
- 综合得分: {accuracy_score:.0%}

## 输出要求
请用中文输出，严格控制在450字符以内。格式如下：
[{ticker}|{trade_date}|得分:{accuracy_score:.0%}]
根因: (预测错误/偏差的根本原因，1-2句)
教训: (核心经验教训，1-2句)
改进: (1-2条具体可执行的改进措施)

不要使用Markdown格式，直接输出纯文本。"""


class AnalysisEvaluator:
    """Evaluates analysis records against actual post-market data.

    Fetches real stock prices, scores the analysis accuracy, and uses
    an LLM to generate structured Lessons Learned that are stored in
    the shared lessons_learned memory namespace.
    """

    def __init__(
        self,
        memory_manager: Optional[object] = None,
        results_dir: Optional[str] = None,
        dry_run: bool = False,
    ):
        """Initialize AnalysisEvaluator.

        Args:
            memory_manager: FinancialMemoryManager for writing lessons.
                            If None, lessons are generated but not stored.
            results_dir: Directory for local tracking files.
            dry_run: If True, skip all side effects (memory writes, record marking).
        """
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
        ticker: Optional[str] = None,
        date: Optional[str] = None,
    ) -> List[str]:
        """Evaluate all pending analysis records from OSS.

        Args:
            ticker: Filter by stock symbol (optional)
            date: Filter by trade date (optional)

        Returns:
            List of generated lesson strings
        """
        today = datetime.now().strftime("%Y-%m-%d")
        pending = await self._record_store.list_pending(before_date=today, ticker=ticker, date=date)

        logger.info("[Evaluator] Found %d pending records", len(pending))

        lessons: List[str] = []
        for record in pending:
            try:
                lesson = await self.evaluate_single(record)
                if lesson:
                    lessons.append(lesson)
            except Exception as e:
                logger.warning(
                    "[Evaluator] Error evaluating %s/%s: %s",
                    record.ticker,
                    record.trade_date,
                    e,
                )

        logger.info("[Evaluator] Generated %d lessons", len(lessons))
        return lessons

    async def evaluate_single(self, record: AnalysisRecord) -> Optional[str]:
        """Evaluate a single analysis record.

        Args:
            record: AnalysisRecord to evaluate

        Returns:
            Generated lesson string, or None on failure
        """
        if not self._ensure_data_imports():
            return None

        # Step 1: Fetch actual stock data
        eval_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.strptime(record.trade_date, "%Y-%m-%d") - timedelta(days=3)).strftime("%Y-%m-%d")
        end_date = (datetime.strptime(eval_date, "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")

        stock_data = self._get_stock_data(record.ticker, start_date, end_date)
        if not stock_data or "Error" in str(stock_data):
            logger.warning("[Evaluator] No stock data for %s", record.ticker)
            return None

        # Step 2: Extract prices
        price_t = _parse_close_price(stock_data, record.trade_date)
        price_tn = _parse_close_price(stock_data, eval_date)

        if price_t is None or price_tn is None:
            logger.warning(
                "[Evaluator] Could not parse prices for %s (T=%s, TN=%s)",
                record.ticker,
                price_t,
                price_tn,
            )
            return None

        # Step 3: Calculate metrics
        actual_return = (price_tn - price_t) / price_t
        direction_correct = (
            (record.direction == "bullish" and actual_return > 0)
            or (record.direction == "bearish" and actual_return < 0)
            or (record.direction == "neutral" and abs(actual_return) < 0.02)
        )

        target_reached = False
        if record.target_price:
            if record.direction == "bullish":
                target_reached = price_tn >= record.target_price
            else:
                target_reached = price_tn <= record.target_price

        stop_loss_triggered = False
        if record.stop_loss:
            if record.direction == "bullish":
                stop_loss_triggered = price_tn <= record.stop_loss
            else:
                stop_loss_triggered = price_tn >= record.stop_loss

        predicted_return = 0.0
        if record.entry_price and record.target_price:
            predicted_return = (record.target_price - record.entry_price) / record.entry_price

        accuracy_score = _calculate_accuracy_score(
            direction_correct=direction_correct,
            target_reached=target_reached,
            stop_loss_triggered=stop_loss_triggered,
            actual_return=actual_return,
            predicted_return=predicted_return,
        )

        # Determine lesson type
        if direction_correct and target_reached:
            lesson_type = "success"
        elif not direction_correct or stop_loss_triggered:
            lesson_type = "failure"
        else:
            lesson_type = "partial"

        # Step 4: Generate lesson via LLM
        lesson_content = await self._generate_lesson(
            record=record,
            eval_date=eval_date,
            price_t=price_t,
            price_tn=price_tn,
            actual_return=actual_return,
            direction_correct=direction_correct,
            target_reached=target_reached,
            stop_loss_triggered=stop_loss_triggered,
            accuracy_score=accuracy_score,
        )

        if not lesson_content:
            return None

        # Step 5: Store lesson in memory
        weight = calculate_lesson_weight(
            lesson_type=lesson_type,
            accuracy_score=accuracy_score,
            confidence=record.confidence,
        )

        if self._memory_manager:
            lessons_mem = self._memory_manager.lessons_memory
            if lessons_mem:
                await lessons_mem.add_reflection_lesson(
                    lesson_content=lesson_content,
                    weight=weight,
                    lesson_type=lesson_type,
                )
                logger.info(
                    "[Evaluator] Stored lesson for %s/%s (type=%s, weight=%.2f)",
                    record.ticker,
                    record.trade_date,
                    lesson_type,
                    weight,
                )

        # Step 6: Mark record as evaluated
        if not self._dry_run:
            self._record_store.mark_evaluated(record.ticker, record.trade_date)

        return lesson_content

    async def _generate_lesson(
        self,
        record: AnalysisRecord,
        eval_date: str,
        price_t: float,
        price_tn: float,
        actual_return: float,
        direction_correct: bool,
        target_reached: bool,
        stop_loss_triggered: bool,
        accuracy_score: float,
    ) -> Optional[str]:
        """Generate a structured lesson using LLM."""
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            logger.warning("[Evaluator] No DASHSCOPE_API_KEY, using template lesson")
            return self._template_lesson(record, accuracy_score, actual_return, direction_correct)

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
            report_excerpt=record.final_decision_summary[:300] or "-",
            eval_date=eval_date,
            price_t=f"${price_t:.2f}",
            price_tn=f"${price_tn:.2f}",
            actual_return=f"{actual_return:+.2%}",
            direction_result="正确" if direction_correct else "错误",
            target_result="是" if target_reached else "否",
            stop_loss_result="是" if stop_loss_triggered else "否",
            accuracy_score=accuracy_score,
        )

        try:
            response = await dashscope.AioGeneration.call(
                api_key=api_key,
                model=DEFAULT_CONFIG["quick_think_llm"],
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
                max_tokens=1024,
                temperature=0.1,
            )
            if response.status_code != 200:
                logger.warning("[Evaluator] DashScope API returned status %s", response.status_code)
                return self._template_lesson(record, accuracy_score, actual_return, direction_correct)

            lesson = (response.output.choices[0].message.content or "").strip()
            # Enforce 500-char limit for Memory API
            return lesson[:500] if lesson else None
        except Exception as e:
            logger.warning("[Evaluator] LLM call failed: %s", e)
            return self._template_lesson(record, accuracy_score, actual_return, direction_correct)

    @staticmethod
    def _template_lesson(
        record: AnalysisRecord,
        accuracy_score: float,
        actual_return: float,
        direction_correct: bool,
    ) -> str:
        """Fallback template-based lesson when LLM is unavailable."""
        direction_text = "看涨" if record.direction == "bullish" else "看跌"
        actual_text = "上涨" if actual_return > 0 else "下跌"
        result_text = "正确" if direction_correct else "错误"

        return (
            f"[{record.ticker}|{record.trade_date}|得分:{accuracy_score:.0%}]\n"
            f"根因: 预测{direction_text}，实际{actual_text}{abs(actual_return):.1%}，"
            f"方向判断{result_text}。{record.reasoning[:60]}\n"
            f"教训: {'继续保持当前分析框架' if direction_correct else '需要增加对反向信号的关注度'}\n"
            f"改进: {'保持并优化止损策略' if direction_correct else '加强反向观点分析权重'}"
        )[:500]


def _parse_close_price(data_str: str, target_date: str) -> Optional[float]:
    """Extract closing price for a target date from CSV-formatted stock data."""
    try:
        # Skip comment lines (starting with #) and empty lines
        lines = [line for line in data_str.strip().split("\n") if line.strip() and not line.strip().startswith("#")]
        if len(lines) < 2:
            return None

        header = lines[0].split(",")
        close_idx = None
        for i, col in enumerate(header):
            if "close" in col.strip().lower():
                close_idx = i
                break
        if close_idx is None:
            return None

        # Exact match
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= close_idx:
                continue
            if target_date in parts[0].strip():
                try:
                    return float(parts[close_idx].strip())
                except ValueError:
                    continue

        # Nearest date within 3 days
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        best_price = None
        best_diff = timedelta(days=999)

        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= close_idx:
                continue
            date_str = parts[0].strip()[:10]
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
                try:
                    line_dt = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                continue

            diff = abs(line_dt - target_dt)
            if diff < best_diff and diff <= timedelta(days=3):
                best_diff = diff
                best_price = float(parts[close_idx].strip())

        return best_price
    except Exception as e:
        logger.warning("Error parsing stock data: %s", e)
        return None


def _calculate_accuracy_score(
    direction_correct: bool,
    target_reached: bool,
    stop_loss_triggered: bool,
    actual_return: float,
    predicted_return: float,
) -> float:
    """Calculate composite accuracy score (0-1).

    Components: direction (40%), target (30%), stop loss (15%), return (15%).
    """
    score = 0.0

    if direction_correct:
        score += 0.4

    if target_reached:
        score += 0.3
    elif direction_correct:
        score += 0.1

    if not stop_loss_triggered:
        score += 0.15

    if predicted_return != 0:
        return_error = abs(actual_return - predicted_return) / max(abs(predicted_return), 0.01)
        score += 0.15 * max(0, 1 - return_error)
    elif actual_return > -0.05:
        score += 0.1

    return min(1.0, max(0.0, score))
