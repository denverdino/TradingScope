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
- 前一交易日收盘价: {price_prev}
- 分析日收盘价: {price_t}
- 实际收益率: {actual_return}
- 方向判断: {direction_result}
- 止损触发: {stop_loss_result}

## 输出要求
请用中文输出，严格控制在450字符以内。格式如下：
[{ticker}|{trade_date}]
根因: (预测错误/偏差的根本原因，1-2句)
教训: (核心经验教训，2-3句)

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
        # Fetch enough history to cover both trade_date and its prior trading day
        start_date = (datetime.strptime(record.trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = (datetime.strptime(record.trade_date, "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")

        stock_data = self._get_stock_data(record.ticker, start_date, end_date)
        if not stock_data or "Error" in str(stock_data):
            logger.warning("[Evaluator] No stock data for %s", record.ticker)
            return None

        # Step 2: Extract prices
        # price_t  : closing price on analysis day (trade_date)
        # price_prev: closing price on the trading day before trade_date
        price_t = _parse_close_price(stock_data, record.trade_date)
        price_prev = _parse_prev_close_price(stock_data, record.trade_date)

        logger.info(
            "[Evaluator] Prices for %s: prev=%s, trade_date=%s at %s",
            record.ticker,
            price_prev,
            price_t,
            record.trade_date,
        )

        if price_t is None or price_prev is None:
            logger.warning(
                "[Evaluator] Could not parse prices for %s (price_t=%s, price_prev=%s)",
                record.ticker,
                price_t,
                price_prev,
            )
            return None

        # Step 3: Calculate metrics using trade_date close vs prior day close
        actual_return = (price_t - price_prev) / price_prev
        direction_correct = (
            (record.direction == "bullish" and actual_return > 0)
            or (record.direction == "bearish" and actual_return < 0)
            or (record.direction == "neutral" and abs(actual_return) < 0.03)
        )

        stop_loss_triggered = False
        if record.stop_loss:
            if record.direction == "bullish":
                stop_loss_triggered = price_t <= record.stop_loss
            elif record.direction == "bearish":
                stop_loss_triggered = price_t >= record.stop_loss
            else:
                # neutral/hold: stop loss triggers when price falls below stop
                stop_loss_triggered = price_t <= record.stop_loss

        # Determine lesson type
        if direction_correct and not stop_loss_triggered:
            lesson_type = "success"
        elif not direction_correct or stop_loss_triggered:
            lesson_type = "failure"
        else:
            lesson_type = "partial"

        # Step 4: Generate lesson via LLM
        lesson_content = await self._generate_lesson(
            record=record,
            price_prev=price_prev,
            price_t=price_t,
            actual_return=actual_return,
            direction_correct=direction_correct,
            stop_loss_triggered=stop_loss_triggered,
        )

        if not lesson_content:
            return None

        # Step 5: Store lesson in memory
        weight = calculate_lesson_weight(lesson_type=lesson_type)

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
        price_prev: float,
        price_t: float,
        actual_return: float,
        direction_correct: bool,
        stop_loss_triggered: bool,
    ) -> Optional[str]:
        """Generate a structured lesson using LLM."""
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            logger.warning("[Evaluator] No DASHSCOPE_API_KEY, skipping lesson generation")
            return None

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
            price_prev=f"${price_prev:.2f}",
            price_t=f"${price_t:.2f}",
            actual_return=f"{actual_return:+.2%}",
            direction_result="正确" if direction_correct else "错误",
            stop_loss_result="是" if stop_loss_triggered else "否",
        )

        logger.info("[Evaluator] Prompt: %s", prompt)

        try:
            response = dashscope.MultiModalConversation.call(
                api_key=api_key,
                model=DEFAULT_CONFIG["deep_think_llm"],
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
                max_tokens=1024,
            )
            if response.status_code != 200:
                logger.warning(
                    "[Evaluator] DashScope API returned status %s: code=%s message=%s",
                    response.status_code,
                    getattr(response, "code", "-"),
                    getattr(response, "message", "-"),
                )
                return None

            lesson = response.output.choices[0].message.content[0]["text"] or ""
            return lesson if lesson else None
        except Exception as e:
            logger.warning("[Evaluator] LLM call failed: %s", e)
            return None


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


def _parse_prev_close_price(data_str: str, trade_date: str) -> Optional[float]:
    """Extract the closing price of the trading day immediately before trade_date.

    Parses all dated rows from the CSV, sorts them, and returns the close price
    of the row whose date is the largest date strictly before trade_date.
    """
    try:
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

        target_dt = datetime.strptime(trade_date, "%Y-%m-%d")
        dated_rows: list = []

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
            if line_dt < target_dt:
                try:
                    dated_rows.append((line_dt, float(parts[close_idx].strip())))
                except ValueError:
                    continue

        if not dated_rows:
            return None

        # Return close price of the most recent day before trade_date
        dated_rows.sort(key=lambda x: x[0])
        return dated_rows[-1][1]
    except Exception as e:
        logger.warning("Error parsing prev close price: %s", e)
        return None
