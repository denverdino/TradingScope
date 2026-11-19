"""Synthetic completed runs and market bars for offline evaluation tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tests.test_output_models import _all_outputs
from tradingscope.agents.evaluation.evaluator import build_evaluation_record
from tradingscope.agents.evaluation.market_outcome import MarketBar
from tradingscope.utils.oss_structured_output_reader import CompletedRun, CompletionManifest

ANALYSIS_DATE = date(2026, 7, 14)
EVALUATED_AT = datetime(2026, 7, 25, tzinfo=timezone.utc)
PREMARKET_RELEASE = datetime(2026, 7, 14, 8, tzinfo=ZoneInfo("America/New_York"))
AFTER_CLOSE_RELEASE = datetime(2026, 7, 14, 16, tzinfo=ZoneInfo("America/New_York"))


def weekdays(start: date, count: int, *, forward: bool = True) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1 if forward else -1)
    return days if forward else list(reversed(days))


def market_bars(
    future: list[tuple[float, float, float, float]],
    *,
    analysis_date: date = ANALYSIS_DATE,
    history: int = 15,
    close: float = 100.0,
) -> list[MarketBar]:
    """Flat history with a 2-point true range (2% ATR at 100), then the given future OHLC bars."""
    history_days = weekdays(analysis_date - timedelta(days=1), history, forward=False)
    bars = [MarketBar(day, close, close + 1, close - 1, close) for day in history_days]
    bars.extend(MarketBar(day, *ohlc) for day, ohlc in zip(weekdays(analysis_date, len(future)), future, strict=True))
    return bars


def flat_future(closes: list[float]) -> list[tuple[float, float, float, float]]:
    """Bars that open at the previous close and trade one point around their close."""
    previous = 100.0
    future = []
    for value in closes:
        future.append((previous, max(previous, value) + 1, min(previous, value) - 1, value))
        previous = value
    return future


def completed_run(
    *,
    run_id: str = "run-1",
    analysis_date: date = ANALYSIS_DATE,
    direction: str = "neutral",
    action: str = "hold",
    analyst_directions: dict[str, str] | None = None,
    portfolio_updates: dict | None = None,
    stages: tuple[str, ...] | None = None,
) -> CompletedRun:
    outputs = {}
    for output in _all_outputs():
        name = output.agent_name
        data = output.model_dump(mode="json")
        data["analysis_date"] = analysis_date.isoformat()
        data["last_completed_session"] = (analysis_date - timedelta(days=1)).isoformat()
        if name in {"research_manager", "trader", "portfolio_manager"}:
            data["decision"].update(direction=direction, action=action)
        if name in {"trader", "portfolio_manager"}:
            data["trade_intent"] = "hold"
        if analyst_directions and name in analyst_directions:
            data["decision"].update(direction=analyst_directions[name], action="hold")
        if name == "portfolio_manager" and portfolio_updates:
            for key, value in portfolio_updates.items():
                if key == "price_plan":
                    data["price_plan"].update(value)
                else:
                    data[key] = value
        if stages is None or name in stages:
            outputs[name] = type(output).model_validate(data)
    manifest = CompletionManifest(
        schema_version="3.1",
        status="complete",
        ticker="AAPL",
        analysis_date=analysis_date,
        run_id=run_id,
        code_version="abc123",
        models={"deep_think_llm": "qwen4-max"},
        last_completed_session=analysis_date - timedelta(days=1),
        forecast_started_at=datetime.combine(analysis_date, datetime.min.time(), tzinfo=timezone.utc),
        forecast_completed_at=datetime.combine(analysis_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(minutes=5),
    )
    return CompletedRun(manifest=manifest, outputs=outputs)


def evaluation_record(horizon: int = 1):
    return build_evaluation_record(
        completed_run(),
        market_bars(flat_future([101.0] * 5)),
        "XLK",
        market_bars(flat_future([100.0] * 5)),
        [],
        horizon,
        evaluated_at=EVALUATED_AT,
    )


class FakeMarketData:
    def __init__(
        self,
        ticker_bars: list[MarketBar],
        benchmark_bars: list[MarketBar] | None = None,
        *,
        benchmark: str = "XLK",
        earnings: list[datetime] | None = None,
    ) -> None:
        self.bars = {"AAPL": ticker_bars, benchmark: benchmark_bars if benchmark_bars is not None else market_bars(flat_future([100.0] * 5))}
        self.benchmark = benchmark
        self.earnings = earnings
        self.calls: list[tuple] = []

    def daily_bars(self, symbol: str, start: str, end: str) -> list[MarketBar]:
        self.calls.append(("bars", symbol, start, end))
        return self.bars[symbol]

    def benchmark_symbol(self, ticker: str) -> str:
        return self.benchmark

    def earnings_dates(self, ticker: str, start: str, end: str) -> list[datetime] | None:
        return self.earnings
