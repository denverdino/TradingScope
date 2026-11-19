"""Benchmark-relative returns and shock-session labelling for one horizon."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Sequence
from zoneinfo import ZoneInfo

from .market_outcome import MarketBar, MarketWindow
from .models import SampleClass, ShockClass, ShockSession


def aligned_benchmark_bars(window: MarketWindow, benchmark_bars: Sequence[MarketBar]) -> list[MarketBar] | None:
    """Return benchmark bars for the reference session and each horizon session, or None if any is missing."""
    by_date = {bar.session_date: bar for bar in benchmark_bars}
    sessions = [window.reference_bar.session_date, *(bar.session_date for bar in window.horizon_bars)]
    aligned = [by_date.get(session) for session in sessions]
    if any(bar is None for bar in aligned):
        return None
    return aligned  # type: ignore[return-value]


def period_return(bars: Sequence[MarketBar]) -> float:
    return bars[-1].close / bars[0].close - 1


def detect_shock_sessions(
    window: MarketWindow,
    benchmark: Sequence[MarketBar],
    *,
    atr_multiple: float,
    earnings_dates: Sequence[datetime] | None,
    ticker_sessions: Sequence[date],
) -> list[ShockSession]:
    """Label horizon sessions whose move or opening gap versus the benchmark exceeds the normal range."""
    limit = atr_multiple * window.daily_atr_pct
    ticker = [window.reference_bar, *window.horizon_bars]
    shocks: list[ShockSession] = []
    for index in range(1, len(ticker)):
        previous, bar = ticker[index - 1], ticker[index]
        bench_previous, bench_bar = benchmark[index - 1], benchmark[index]
        excess_return = (bar.close / previous.close - 1) - (bench_bar.close / bench_previous.close - 1)
        excess_gap = (bar.open / previous.close - 1) - (bench_bar.open / bench_previous.close - 1)
        if abs(excess_return) > limit or abs(excess_gap) > limit:
            shocks.append(
                ShockSession(
                    session_date=bar.session_date,
                    excess_return=excess_return,
                    excess_gap=excess_gap,
                    classification=_classify(bar.session_date, earnings_dates, ticker_sessions),
                )
            )
    return shocks


def sample_class(shocks: Sequence[ShockSession]) -> SampleClass:
    """Scheduled shocks were knowable, so only unscheduled or unknown shocks exclude a sample."""
    if any(shock.classification is not ShockClass.SCHEDULED for shock in shocks):
        return SampleClass.SHOCK_EXCLUDED
    return SampleClass.CLEAN


def _classify(session: date, earnings_dates: Sequence[datetime] | None, ticker_sessions: Sequence[date]) -> ShockClass:
    if earnings_dates is None:
        return ShockClass.UNKNOWN
    ordered = sorted(ticker_sessions)
    uncertain = False
    for release in earnings_dates:
        if not isinstance(release, datetime) or release.tzinfo is None:
            release_date = release.date() if isinstance(release, datetime) else release
            next_session = next((candidate for candidate in ordered if candidate > release_date), None)
            uncertain |= session in {release_date, next_session}
            continue
        release = release.astimezone(ZoneInfo("America/New_York"))
        if time(9, 30) <= release.time() < time(16):
            uncertain |= session == release.date()
            continue
        after_close = release.time() >= time(16)
        reaction = next(
            (candidate for candidate in ordered if candidate > release.date() or (candidate == release.date() and not after_close)),
            None,
        )
        if session == reaction:
            return ShockClass.SCHEDULED
    return ShockClass.UNKNOWN if uncertain else ShockClass.UNSCHEDULED
