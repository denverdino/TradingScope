"""Deterministic OHLC assessment for persisted trading decisions."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from io import StringIO
from typing import Literal, Sequence

from .models import AnalysisRecord


@dataclass(frozen=True)
class MarketBar:
    """One completed daily OHLC market bar."""

    trade_date: date
    open: float
    high: float
    low: float
    close: float


class AssessmentStatus(StrEnum):
    """Deterministic outcome of a directional or executable assessment."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    INCONCLUSIVE = "inconclusive"
    NOT_FILLED = "not_filled"


@dataclass(frozen=True)
class TradeAssessment:
    """Observed market outcome over one supported future-session horizon."""

    horizon_days: int
    status: AssessmentStatus
    entry_triggered: bool
    entry_price: float | None
    exit_reason: str | None
    benchmark_return: float
    strategy_return: float | None
    atr_threshold: float
    limitations: tuple[str, ...]


def parse_market_bars(csv_text: str) -> list[MarketBar]:
    """Parse vendor CSV into ascending, unique daily OHLC bars."""
    reader = csv.DictReader(StringIO(csv_text))
    if reader.fieldnames is None:
        raise ValueError("market CSV requires a header")

    fields = {name.strip().lower(): name for name in reader.fieldnames}
    date_field = next((fields[name] for name in ("timestamp", "date", "trade_date") if name in fields), None)
    required = {name: fields.get(name) for name in ("open", "high", "low", "close")}
    if date_field is None or any(field is None for field in required.values()):
        raise ValueError("market CSV requires a date and OHLC columns")

    bars: list[MarketBar] = []
    seen_dates: set[date] = set()
    for row in reader:
        try:
            trade_date = date.fromisoformat(row[date_field].strip())
            raw_prices = {name: row[field] for name, field in required.items() if field is not None}
            if all(value is not None and not value.strip() for value in raw_prices.values()):
                continue
            prices = {name: float(value) for name, value in raw_prices.items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid market bar") from exc
        if any(not math.isfinite(value) or value <= 0 for value in prices.values()):
            raise ValueError("market bar price must be finite and positive")
        if trade_date in seen_dates:
            raise ValueError(f"duplicate market date: {trade_date.isoformat()}")
        seen_dates.add(trade_date)
        bars.append(MarketBar(trade_date=trade_date, **prices))

    return sorted(bars, key=lambda bar: bar.trade_date)


def assess_trade(
    record: AnalysisRecord,
    bars: Sequence[MarketBar],
    horizon_days: Literal[1, 3, 5],
) -> TradeAssessment | None:
    """Assess a record without network calls or inferred future prices."""
    if type(horizon_days) is not int or horizon_days not in (1, 3, 5):
        raise ValueError(f"Invalid evaluation horizon {horizon_days}; allowed horizons are 1, 3, 5.")

    analysis_date = date.fromisoformat(record.trade_date)
    ordered_bars = sorted(bars, key=lambda bar: bar.trade_date)
    base_bar = next((bar for bar in ordered_bars if bar.trade_date == analysis_date), None)
    history = [bar for bar in ordered_bars if bar.trade_date < analysis_date]
    future = [bar for bar in ordered_bars if bar.trade_date > analysis_date]
    if base_bar is None or len(history) < 15 or len(future) < horizon_days:
        return None

    atr = _average_true_range(history[-15:])
    atr_threshold = atr / base_bar.close * math.sqrt(horizon_days)
    horizon_bar = future[horizon_days - 1]
    benchmark_return = (horizon_bar.close - base_bar.close) / base_bar.close
    status = _direction_status(record.direction, benchmark_return, atr_threshold)

    if record.trade_intent is None or record.intent_inferred:
        return _direction_only_assessment(
            horizon_days,
            status,
            benchmark_return,
            atr_threshold,
            "No explicit historical trade intent; direction-only assessment.",
        )
    if record.trade_intent == "hold":
        return _direction_only_assessment(
            horizon_days,
            status,
            benchmark_return,
            atr_threshold,
            "Hold intent produces no new execution simulation.",
        )

    future_window = future[:horizon_days]
    if record.trade_intent in {"open_long", "open_short"} and record.time_stop_days is not None:
        fill_window = future_window[: record.time_stop_days]
    else:
        fill_window = future_window
    limitations = (
        ("The residual-position time stop for reduce_long is not simulated.",)
        if record.trade_intent == "reduce_long" and record.time_stop_days is not None
        else ()
    )
    fill = _first_fill(record, fill_window)
    if fill is None:
        return TradeAssessment(
            horizon_days=horizon_days,
            status=AssessmentStatus.NOT_FILLED,
            entry_triggered=False,
            entry_price=None,
            exit_reason="not_filled",
            benchmark_return=benchmark_return,
            strategy_return=None,
            atr_threshold=atr_threshold,
            limitations=limitations,
        )

    fill_index, execution_price = fill
    if record.trade_intent in {"open_long", "open_short"}:
        strategy_return, exit_reason = _simulate_open_trade(
            record,
            future_window,
            fill_index,
            execution_price,
        )
    elif record.trade_intent in {"reduce_long", "close_long"}:
        strategy_return = (execution_price - horizon_bar.close) / execution_price
        exit_reason = "execution"
    elif record.trade_intent == "cover_short":
        strategy_return = (horizon_bar.close - execution_price) / execution_price
        exit_reason = "execution"
    else:
        return _direction_only_assessment(
            horizon_days,
            status,
            benchmark_return,
            atr_threshold,
            f"Unsupported trade intent {record.trade_intent!r}; direction-only assessment.",
        )

    return TradeAssessment(
        horizon_days=horizon_days,
        status=_strategy_status(strategy_return, atr_threshold),
        entry_triggered=True,
        entry_price=execution_price,
        exit_reason=exit_reason,
        benchmark_return=benchmark_return,
        strategy_return=strategy_return,
        atr_threshold=atr_threshold,
        limitations=limitations,
    )


def _direction_only_assessment(
    horizon_days: int,
    status: AssessmentStatus,
    benchmark_return: float,
    atr_threshold: float,
    limitation: str,
) -> TradeAssessment:
    return TradeAssessment(
        horizon_days=horizon_days,
        status=status,
        entry_triggered=False,
        entry_price=None,
        exit_reason=None,
        benchmark_return=benchmark_return,
        strategy_return=None,
        atr_threshold=atr_threshold,
        limitations=(limitation,),
    )


def _first_fill(
    record: AnalysisRecord,
    future_bars: Sequence[MarketBar],
) -> tuple[int, float] | None:
    if record.entry_price is None:
        return None
    range_low = record.entry_price_low if record.entry_price_low is not None else record.entry_price
    range_high = record.entry_price_high if record.entry_price_high is not None else record.entry_price
    for index, bar in enumerate(future_bars):
        intersection_low = max(range_low, bar.low)
        intersection_high = min(range_high, bar.high)
        if intersection_low <= intersection_high:
            execution_price = min(max(record.entry_price, intersection_low), intersection_high)
            return index, execution_price
    return None


def _simulate_open_trade(
    record: AnalysisRecord,
    future_bars: Sequence[MarketBar],
    fill_index: int,
    execution_price: float,
) -> tuple[float, str]:
    is_long = record.trade_intent == "open_long"
    for index in range(fill_index, len(future_bars)):
        bar = future_bars[index]
        stop_hit = record.stop_loss is not None and (bar.low <= record.stop_loss if is_long else bar.high >= record.stop_loss)
        target_hit = record.target_price is not None and (bar.high >= record.target_price if is_long else bar.low <= record.target_price)
        if stop_hit:
            return _directional_return(is_long, execution_price, record.stop_loss), "stop_loss"
        if target_hit:
            return _directional_return(is_long, execution_price, record.target_price), "target"
        if record.time_stop_days is not None and index + 1 >= record.time_stop_days:
            return _directional_return(is_long, execution_price, bar.close), "time_stop"

    return _directional_return(is_long, execution_price, future_bars[-1].close), "horizon"


def _directional_return(is_long: bool, entry_price: float, exit_price: float) -> float:
    if is_long:
        return (exit_price - entry_price) / entry_price
    return (entry_price - exit_price) / entry_price


def _strategy_status(strategy_return: float, threshold: float) -> AssessmentStatus:
    if strategy_return > threshold:
        return AssessmentStatus.CORRECT
    if strategy_return < -threshold:
        return AssessmentStatus.INCORRECT
    return AssessmentStatus.INCONCLUSIVE


def _average_true_range(history: Sequence[MarketBar]) -> float:
    true_ranges: list[float] = []
    previous_close = history[0].close
    for bar in history[1:]:
        candidates = [
            bar.high - bar.low,
            abs(bar.high - previous_close),
            abs(bar.low - previous_close),
        ]
        true_ranges.append(max(candidates))
        previous_close = bar.close
    return sum(true_ranges) / len(true_ranges)


def _direction_status(direction: str, actual_return: float, threshold: float) -> AssessmentStatus:
    if direction == "bullish":
        if actual_return > threshold:
            return AssessmentStatus.CORRECT
        if actual_return < -threshold:
            return AssessmentStatus.INCORRECT
        return AssessmentStatus.INCONCLUSIVE
    if direction == "bearish":
        if actual_return < -threshold:
            return AssessmentStatus.CORRECT
        if actual_return > threshold:
            return AssessmentStatus.INCORRECT
        return AssessmentStatus.INCONCLUSIVE
    if -threshold <= actual_return <= threshold:
        return AssessmentStatus.CORRECT
    return AssessmentStatus.INCORRECT
