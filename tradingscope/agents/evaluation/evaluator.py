"""Deterministic post-market evaluation of completed analysis runs.

Every stage output of a completed run is judged at each configured horizon. Records are
written once per run and horizon; no model is called.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from functools import partial
from typing import Callable, Iterable, Protocol
from zoneinfo import ZoneInfo

from tradingscope.agents.output import PortfolioManagerOutput, TraderOutput
from tradingscope.utils.oss_structured_output_reader import (
    STAGE_NAMES,
    CompletedRun,
    CompletionManifest,
    fetch_completed_manifest,
    fetch_completed_run,
)

from .market_outcome import EVALUATION_CONFIG, MarketBar, assess_trade_in_window, direction_status, market_window, parse_market_bars
from .models import AnalysisRecord, EvaluationRecord, MarketOutcome, StageEvaluation, TradeOutcome
from .oss_store import EvaluationStore, evaluation_key
from .shocks import aligned_benchmark_bars, detect_shock_sessions, period_return, sample_class

logger = logging.getLogger(__name__)


class MarketData(Protocol):
    def daily_bars(self, symbol: str, start: str, end: str) -> list[MarketBar]: ...

    def benchmark_symbol(self, ticker: str) -> str: ...

    def earnings_dates(self, ticker: str, start: str, end: str) -> list[datetime] | None: ...


class VendorMarketData:
    """Market data through the configured vendors; earnings and sectors come from yfinance."""

    def daily_bars(self, symbol: str, start: str, end: str) -> list[MarketBar]:
        from tradingscope.dataflows.interface import route_to_vendor

        csv_text = route_to_vendor("get_stock_data", symbol, start, end)
        if not csv_text or "Error" in str(csv_text):
            raise ValueError(f"no market data for {symbol}")
        return parse_market_bars(_strip_vendor_preamble(csv_text))

    def benchmark_symbol(self, ticker: str) -> str:
        from tradingscope.dataflows.y_finance import get_sector_benchmark_symbol

        return get_sector_benchmark_symbol(ticker)

    def earnings_dates(self, ticker: str, start: str, end: str) -> list[datetime] | None:
        from tradingscope.dataflows.y_finance import fetch_earnings_dates

        return fetch_earnings_dates(ticker, start, end)


def trade_record(output: TraderOutput | PortfolioManagerOutput) -> AnalysisRecord:
    """Build the simulation input from a stage output that carries a trade plan."""
    if output.trade_intent is None:
        raise ValueError("evaluation requires an explicit trade intent")
    return AnalysisRecord(
        analysis_date=output.analysis_date.isoformat(),
        direction=output.decision.direction.value,
        execution_price=output.price_plan.execution_price,
        execution_price_low=output.price_plan.execution_price_low,
        execution_price_high=output.price_plan.execution_price_high,
        target_price=output.price_plan.target_price,
        stop_loss=output.price_plan.stop_loss,
        trade_intent=output.trade_intent.value,
        position_advice=output.position_advice.value,
        time_stop_days=output.time_stop_days,
    )


def build_evaluation_record(
    run: CompletedRun,
    bars: list[MarketBar],
    benchmark_symbol: str,
    benchmark_bars: list[MarketBar],
    earnings_dates: list[datetime] | None,
    horizon: int,
    *,
    evaluated_at: datetime,
) -> EvaluationRecord | None:
    """Judge every stage output of a run at one horizon, or None until data is sufficient."""
    analysis_date = run.manifest.analysis_date
    known_session = run.manifest.last_completed_session
    if known_session >= analysis_date:
        raise ValueError("analysis date must follow the last completed session; already-observed sessions cannot be scored")
    window = market_window(bars, analysis_date, horizon)
    if window is None:
        return None
    first_open = datetime.combine(window.horizon_bars[0].session_date, time(9, 30), tzinfo=ZoneInfo("America/New_York"))
    if run.manifest.forecast_completed_at >= first_open:
        raise ValueError("daily OHLC cannot score a forecast completed after its first session opened")
    if window.horizon_end.session_date >= evaluated_at.astimezone(timezone.utc).date():
        return None
    benchmark = aligned_benchmark_bars(window, benchmark_bars)
    if benchmark is None:
        return None

    threshold = window.threshold()
    underlying_return = window.horizon_end.close / window.reference_bar.close - 1
    benchmark_return = period_return(benchmark)
    excess_return = underlying_return - benchmark_return
    shocks = detect_shock_sessions(
        window,
        benchmark,
        atr_multiple=EVALUATION_CONFIG["shock_atr_multiple"],
        earnings_dates=earnings_dates,
        ticker_sessions=[bar.session_date for bar in bars],
    )

    stages = []
    for name in STAGE_NAMES:
        output = run.outputs.get(name)
        if output is None:
            continue
        direction = output.decision.direction.value
        trade = None
        if isinstance(output, (TraderOutput, PortfolioManagerOutput)):
            assessment = assess_trade_in_window(trade_record(output), window)
            trade = TradeOutcome(
                trade_intent=output.trade_intent.value,
                simulated=assessment.exit_reason is not None,
                status=assessment.status.value,
                filled=assessment.filled,
                fill_price=assessment.fill_price,
                exit_reason=assessment.exit_reason,
                strategy_return=assessment.strategy_return,
                strategy_return_basis=assessment.strategy_return_basis,
                residual_exit_reason=assessment.residual_exit_reason,
                residual_exit_date=assessment.residual_exit_date,
                residual_exit_price=assessment.residual_exit_price,
                limitations=list(assessment.limitations),
            )
        stages.append(
            StageEvaluation(
                stage=name,
                direction=direction,
                confidence=output.decision.confidence,
                direction_status=direction_status(direction, underlying_return, threshold).value,
                excess_direction_status=direction_status(direction, excess_return, threshold).value,
                trade=trade,
            )
        )

    return EvaluationRecord(
        run_id=run.run_id,
        code_version=run.manifest.code_version,
        models=run.manifest.models,
        ticker=run.manifest.ticker,
        analysis_date=analysis_date,
        horizon=horizon,
        horizon_end_session=window.horizon_end.session_date,
        evaluated_at=evaluated_at,
        forecast_started_at=run.manifest.forecast_started_at,
        forecast_completed_at=run.manifest.forecast_completed_at,
        market=MarketOutcome(
            reference_session=window.reference_bar.session_date,
            reference_close=window.reference_bar.close,
            daily_atr_pct=window.daily_atr_pct,
            atr_threshold=threshold,
            underlying_return=underlying_return,
            benchmark_symbol=benchmark_symbol,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            shock_sessions=shocks,
            sample_class=sample_class(shocks),
        ),
        stages=stages,
        source_outputs={name: output.model_dump(mode="json") for name, output in run.outputs.items()},
    )


class AnalysisEvaluator:
    """Find completed runs with pending horizons, judge them, and store each record once."""

    def __init__(
        self,
        store: EvaluationStore,
        *,
        market_data: MarketData | None = None,
        fetch_run: Callable[[str, str], CompletedRun | None] | None = None,
        results_dir: str | None = None,
        dry_run: bool = False,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._store = store
        self._market_data = market_data or VendorMarketData()
        self._fetch_run = fetch_run or partial(fetch_completed_run, results_dir=results_dir)
        self._dry_run = dry_run
        self._clock = clock

    def candidates(self, tickers: list[str], *, analysis_date: str | None) -> list[tuple[str, str]]:
        today = datetime.now().date()
        if analysis_date:
            dates = [analysis_date]
        else:
            lookback = EVALUATION_CONFIG["lookback_calendar_days"]
            dates = [(today - timedelta(days=offset)).isoformat() for offset in range(lookback, 0, -1)]
        return [(value, ticker) for value in dates for ticker in tickers]

    def run(self, tickers: list[str], *, analysis_date: str | None = None) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        for candidate_date, ticker in self.candidates(tickers, analysis_date=analysis_date):
            try:
                records.extend(self._evaluate_candidate(candidate_date, ticker))
            except Exception as exc:
                logger.warning("[Evaluator] Error evaluating %s/%s: %s", ticker, candidate_date, exc)
        logger.info("[Evaluator] Produced %d evaluation records", len(records))
        return records

    def _evaluate_candidate(self, analysis_date: str, ticker: str) -> list[EvaluationRecord]:
        run = self._fetch_run(analysis_date, ticker)
        if run is None:
            return []
        pending = [horizon for horizon in EVALUATION_CONFIG["horizons"] if not self._store.exists(evaluation_key(analysis_date, ticker, run.run_id, horizon))]
        if not pending:
            return []

        analysis_day = date.fromisoformat(analysis_date)
        start = (analysis_day - timedelta(days=EVALUATION_CONFIG["price_history_calendar_days"])).isoformat()
        end = (analysis_day + timedelta(days=EVALUATION_CONFIG["price_future_calendar_days"])).isoformat()
        bars = self._market_data.daily_bars(ticker, start, end)
        benchmark_symbol = self._market_data.benchmark_symbol(ticker)
        benchmark_bars = self._market_data.daily_bars(benchmark_symbol, start, end)
        earnings_dates = self._market_data.earnings_dates(ticker, start, end)

        records = []
        for horizon in pending:
            record = build_evaluation_record(
                run,
                bars,
                benchmark_symbol,
                benchmark_bars,
                earnings_dates,
                horizon,
                evaluated_at=self._clock(),
            )
            if record is None:
                logger.info("[Evaluator] %s/%s/h%s pending: insufficient ticker or benchmark data", ticker, analysis_date, horizon)
                continue
            if not self._dry_run:
                self._store.write(record)
            records.append(record)
        return records


def current_records(
    records: Iterable[EvaluationRecord],
    fetch_manifest: Callable[[str, str], CompletionManifest | None] = fetch_completed_manifest,
) -> list[EvaluationRecord]:
    """Drop records of runs superseded by a later run of the same ticker and date."""
    groups: dict[tuple[date, str], list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        if record.horizon in EVALUATION_CONFIG["horizons"]:
            groups[(record.analysis_date, record.ticker)].append(record)
    kept = []
    for (analysis_date, ticker), group in groups.items():
        manifest = fetch_manifest(analysis_date.isoformat(), ticker)
        if manifest is None:
            continue
        current = manifest.run_id
        kept.extend(record for record in group if record.run_id == current)
    return kept


def _strip_vendor_preamble(csv_text: str) -> str:
    """Remove yfinance metadata lines before strict CSV parsing."""
    return "\n".join(line for line in csv_text.splitlines() if line.strip() and not line.lstrip().startswith("#"))
