"""Prediction Evaluator for the Reflection Loop system.

This module evaluates predictions against actual stock prices:
- Fetches actual prices at T day and T+N day
- Calculates return, direction correctness, target reached, etc.
- Generates accuracy scores for reflection analysis
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Optional

from agentscope import logger

from .models import EvaluationResult, PredictionRecord


def _parse_stock_data(data_str: str, target_date: str) -> Optional[float]:
    """Parse stock data string to extract closing price for target date.

    Args:
        data_str: CSV-formatted stock data string
        target_date: Date to extract price for (YYYY-MM-DD)

    Returns:
        Closing price for target date, or None if not found
    """
    try:
        lines = data_str.strip().split("\n")
        if len(lines) < 2:
            return None

        # Find header row and Close column index
        header = lines[0].split(",")
        close_idx = None
        date_idx = 0  # Date is typically first column

        for i, col in enumerate(header):
            col_lower = col.strip().lower()
            if col_lower == "close" or col_lower == "adj close":
                close_idx = i
                break

        if close_idx is None:
            # Try to find by pattern
            for i, col in enumerate(header):
                if "close" in col.lower():
                    close_idx = i
                    break

        if close_idx is None:
            return None

        # Search for target date
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= close_idx:
                continue

            date_str = parts[date_idx].strip()
            # Handle various date formats
            if target_date in date_str or date_str.startswith(target_date):
                try:
                    return float(parts[close_idx].strip())
                except ValueError:
                    continue

        # If exact date not found, try to find closest date
        # (market might be closed on target date)
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        best_price = None
        best_diff = timedelta(days=999)

        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= close_idx:
                continue

            date_str = parts[date_idx].strip()
            try:
                # Try common date formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"]:
                    try:
                        line_dt = datetime.strptime(date_str[:10], fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue

                diff = abs(line_dt - target_dt)
                if diff < best_diff and diff <= timedelta(days=3):
                    best_diff = diff
                    best_price = float(parts[close_idx].strip())

            except (ValueError, IndexError):
                continue

        return best_price

    except Exception as e:
        logger.warning(f"Error parsing stock data: {e}")
        return None


def _parse_market_return(indices_str: str) -> float:
    """Parse market indices data to extract S&P 500 return.

    Args:
        indices_str: Formatted market indices string

    Returns:
        S&P 500 period return as decimal (e.g., 0.02 for 2%)
    """
    try:
        # Look for S&P 500 return in the data
        # Format varies, try multiple patterns
        patterns = [
            r"S&P\s*500.*?(\-?\d+\.?\d*)%",
            r"SPY.*?(\-?\d+\.?\d*)%",
            r"\^GSPC.*?(\-?\d+\.?\d*)%",
        ]

        for pattern in patterns:
            match = re.search(pattern, indices_str, re.IGNORECASE)
            if match:
                return float(match.group(1)) / 100

        return 0.0

    except Exception:
        return 0.0


class PredictionEvaluator:
    """Evaluates predictions against actual stock performance.

    This class fetches actual stock prices and calculates various
    metrics to determine prediction accuracy.

    Usage:
        evaluator = PredictionEvaluator()
        result = await evaluator.evaluate(prediction)
    """

    def __init__(self):
        """Initialize PredictionEvaluator."""
        # Import data functions lazily to avoid circular imports
        self._get_stock_data = None
        self._get_market_indices = None

    def _ensure_imports(self) -> bool:
        """Ensure data fetching functions are imported."""
        if self._get_stock_data is not None:
            return True

        try:
            from tradingscope.dataflows.interface import route_to_vendor

            def get_stock_data(symbol, start_date, end_date):
                return route_to_vendor("get_stock_data", symbol, start_date, end_date)

            def get_market_indices(look_back_days):
                return route_to_vendor("get_market_indices", look_back_days)

            self._get_stock_data = get_stock_data
            self._get_market_indices = get_market_indices
            return True

        except ImportError as e:
            logger.warning(f"Failed to import data functions: {e}")
            return False

    async def evaluate(self, prediction: PredictionRecord) -> Optional[EvaluationResult]:
        """Evaluate a prediction against actual stock performance.

        Args:
            prediction: PredictionRecord to evaluate

        Returns:
            EvaluationResult with calculated metrics, or None if evaluation fails
        """
        if not self._ensure_imports():
            return None

        try:
            # Calculate date range for data fetch
            pred_date = datetime.strptime(prediction.prediction_date, "%Y-%m-%d")
            eval_date = datetime.strptime(prediction.evaluation_date, "%Y-%m-%d")

            # Extend range by a few days to handle market closures
            start_date = (pred_date - timedelta(days=5)).strftime("%Y-%m-%d")
            end_date = (eval_date + timedelta(days=5)).strftime("%Y-%m-%d")

            # Fetch stock data
            stock_data = self._get_stock_data(prediction.symbol, start_date, end_date)

            if not stock_data or "Error" in stock_data:
                logger.warning(f"Failed to fetch stock data for {prediction.symbol}: {stock_data}")
                return None

            # Extract prices
            actual_price_t = _parse_stock_data(stock_data, prediction.prediction_date)
            actual_price_tn = _parse_stock_data(stock_data, prediction.evaluation_date)

            if actual_price_t is None or actual_price_tn is None:
                logger.warning(f"Could not find prices for {prediction.symbol} at {prediction.prediction_date} or {prediction.evaluation_date}")
                return None

            # Calculate returns
            actual_return = (actual_price_tn - actual_price_t) / actual_price_t

            # Calculate predicted return if entry/target prices available
            predicted_return = 0.0
            if prediction.entry_price and prediction.target_price:
                predicted_return = (prediction.target_price - prediction.entry_price) / prediction.entry_price

            # Determine direction correctness
            actual_direction = "bullish" if actual_return > 0.001 else ("bearish" if actual_return < -0.001 else "neutral")
            direction_correct = (
                (prediction.direction == "bullish" and actual_return > 0)
                or (prediction.direction == "bearish" and actual_return < 0)
                or (prediction.direction == "neutral" and abs(actual_return) < 0.02)
            )

            # Check if target was reached
            target_reached = False
            if prediction.target_price:
                if prediction.direction == "bullish":
                    # For bullish, check if price went above target
                    target_reached = actual_price_tn >= prediction.target_price
                else:
                    # For bearish, check if price went below target
                    target_reached = actual_price_tn <= prediction.target_price

            # Check if stop loss was triggered
            stop_loss_triggered = False
            if prediction.stop_loss:
                if prediction.direction == "bullish":
                    stop_loss_triggered = actual_price_tn <= prediction.stop_loss
                else:
                    stop_loss_triggered = actual_price_tn >= prediction.stop_loss

            # Calculate accuracy score (0-1)
            accuracy_score = self._calculate_accuracy_score(
                direction_correct=direction_correct,
                target_reached=target_reached,
                stop_loss_triggered=stop_loss_triggered,
                actual_return=actual_return,
                predicted_return=predicted_return,
                confidence=prediction.confidence,
            )

            # Get market return for comparison
            days_diff = (eval_date - pred_date).days
            try:
                market_data = self._get_market_indices(days_diff + 10)
                market_return = _parse_market_return(market_data)
            except Exception:
                market_return = 0.0

            return EvaluationResult(
                prediction_id=prediction.prediction_id,
                actual_price_t=actual_price_t,
                actual_price_tn=actual_price_tn,
                actual_return=actual_return,
                predicted_return=predicted_return,
                direction_correct=direction_correct,
                target_reached=target_reached,
                stop_loss_triggered=stop_loss_triggered,
                accuracy_score=accuracy_score,
                market_return=market_return,
                evaluation_date=datetime.now().strftime("%Y-%m-%d"),
            )

        except Exception as e:
            logger.warning(f"Error evaluating prediction {prediction.prediction_id}: {e}")
            return None

    def _calculate_accuracy_score(
        self,
        direction_correct: bool,
        target_reached: bool,
        stop_loss_triggered: bool,
        actual_return: float,
        predicted_return: float,
        confidence: float,
    ) -> float:
        """Calculate comprehensive accuracy score.

        Scoring components:
        - Direction correctness: 40%
        - Target achievement: 30%
        - Stop loss avoidance: 15%
        - Return accuracy: 15%

        Args:
            Various evaluation metrics

        Returns:
            Accuracy score between 0 and 1
        """
        score = 0.0

        # Direction component (40%)
        if direction_correct:
            score += 0.4

        # Target component (30%)
        if target_reached:
            score += 0.3
        elif direction_correct:
            # Partial credit if direction was right
            score += 0.1

        # Stop loss component (15%)
        if not stop_loss_triggered:
            score += 0.15

        # Return accuracy component (15%)
        if predicted_return != 0:
            return_error = abs(actual_return - predicted_return) / max(abs(predicted_return), 0.01)
            return_accuracy = max(0, 1 - return_error)
            score += 0.15 * return_accuracy
        else:
            # No prediction, give partial credit if small loss
            if actual_return > -0.05:
                score += 0.1

        return min(1.0, max(0.0, score))
