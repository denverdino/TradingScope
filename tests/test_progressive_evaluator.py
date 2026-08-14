from __future__ import annotations

import asyncio
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from tradingscope.agents.evaluation.evaluator import (
    AnalysisEvaluator,
    _lesson_type_for_status,
)
from tradingscope.agents.evaluation.market_outcome import AssessmentStatus
from tradingscope.agents.evaluation.models import AnalysisRecord, EvaluationResult


def _record(ticker: str = "AAPL", trade_date: str = "2025-01-21") -> AnalysisRecord:
    return AnalysisRecord(
        ticker=ticker,
        trade_date=trade_date,
        direction="bullish",
        action="buy",
        confidence=0.8,
        reasoning="盈利预期改善",
        trade_intent=None,
        intent_inferred=True,
    )


def _market_csv(
    *,
    analysis_day: date = date(2025, 1, 21),
    history_days: int = 15,
    future_days: int = 5,
) -> str:
    prior_days: list[date] = []
    cursor = analysis_day - timedelta(days=1)
    while len(prior_days) < history_days:
        if cursor.weekday() < 5:
            prior_days.append(cursor)
        cursor -= timedelta(days=1)
    prior_days.reverse()

    evaluation_days: list[date] = []
    cursor = analysis_day
    while len(evaluation_days) < future_days:
        if cursor.weekday() < 5:
            evaluation_days.append(cursor)
        cursor += timedelta(days=1)

    rows = ["timestamp,open,high,low,close"]
    for day in prior_days:
        rows.append(f"{day.isoformat()},100,102,98,100")
    for index, day in enumerate(evaluation_days, start=1):
        close = 100 + index * 5
        rows.append(f"{day.isoformat()},100,{close + 1},99,{close}")
    return "\n".join(rows)


class _FakeStore:
    def __init__(self, completed: set[int] | None = None) -> None:
        self.completed = completed or set()
        self.marked: list[tuple[str, str, int]] = []
        self.records: list[AnalysisRecord] = []

    async def list_pending(self, **_kwargs) -> list[AnalysisRecord]:
        return self.records

    def is_evaluated(self, _ticker: str, _trade_date: str, horizon_days: int) -> bool:
        return horizon_days in self.completed

    def mark_evaluated(self, ticker: str, trade_date: str, horizon_days: int) -> bool:
        self.marked.append((ticker, trade_date, horizon_days))
        return True


def _evaluator(
    csv_text: str,
    *,
    completed: set[int] | None = None,
    dry_run: bool = False,
) -> tuple[AnalysisEvaluator, _FakeStore]:
    evaluator = AnalysisEvaluator(model=SimpleNamespace(), dry_run=dry_run)
    store = _FakeStore(completed)
    evaluator._record_store = store
    evaluator._get_stock_data = lambda *_args: csv_text
    evaluator._generate_lesson = AsyncMock(
        return_value="评估结果: 客观状态得到解释\n经验教训: 保持纪律",
    )
    return evaluator, store


def test_only_mature_pending_horizons_are_assessed_and_marked() -> None:
    evaluator, store = _evaluator(_market_csv(future_days=2))

    results = asyncio.run(evaluator.evaluate_single(_record()))

    assert [result.horizon_days for result in results] == [1]
    assert store.marked == [("AAPL", "2025-01-21", 1)]


def test_weekend_record_uses_friday_close_and_monday_for_horizon_one() -> None:
    weekend = date(2025, 1, 25)
    evaluator, store = _evaluator(
        _market_csv(analysis_day=weekend, future_days=1),
    )

    results = asyncio.run(
        evaluator.evaluate_single(_record(trade_date=weekend.isoformat())),
    )

    assert [result.horizon_days for result in results] == [1]
    assert store.marked == [("AAPL", "2025-01-25", 1)]


def test_unavailable_horizons_are_logged_and_left_pending() -> None:
    evaluator, store = _evaluator(_market_csv(future_days=1))

    with patch("tradingscope.agents.evaluation.evaluator.logger") as logger:
        results = asyncio.run(evaluator.evaluate_single(_record()))

    assert [result.horizon_days for result in results] == [1]
    assert store.marked == [("AAPL", "2025-01-21", 1)]
    message = "[Evaluator] Skipping %s/%s/%s: insufficient market data (history=%s, required_history=15, evaluation_sessions=%s, required_sessions=%s)"
    logger.info.assert_any_call(message, "AAPL", "2025-01-21", 3, 15, 1, 3)
    logger.info.assert_any_call(message, "AAPL", "2025-01-21", 5, 15, 1, 5)


def test_yfinance_comment_preamble_is_accepted() -> None:
    csv_text = "# Stock data for AAPL\n# Total records: 20\n\n" + _market_csv()
    evaluator, _store = _evaluator(csv_text)

    results = asyncio.run(evaluator.evaluate_single(_record()))

    assert [result.horizon_days for result in results] == [1, 3, 5]


def test_market_data_is_fetched_once_for_the_full_progressive_window() -> None:
    evaluator, _store = _evaluator(_market_csv())
    get_stock_data = Mock(return_value=_market_csv())
    evaluator._get_stock_data = get_stock_data

    asyncio.run(evaluator.evaluate_single(_record()))

    get_stock_data.assert_called_once_with("AAPL", "2024-12-17", "2025-02-04")


def test_completed_horizons_are_skipped() -> None:
    evaluator, store = _evaluator(_market_csv(), completed={1})

    results = asyncio.run(evaluator.evaluate_single(_record()))

    assert [result.horizon_days for result in results] == [3, 5]
    assert [marked[2] for marked in store.marked] == [3, 5]


def test_all_three_horizons_produce_separate_objective_results() -> None:
    evaluator, _store = _evaluator(_market_csv())

    results = asyncio.run(evaluator.evaluate_single(_record()))

    assert [result.horizon_days for result in results] == [1, 3, 5]
    assert [result.trade_date for result in results] == ["2025-01-21"] * 3
    assert all(result.status == AssessmentStatus.CORRECT for result in results)
    assert [result.benchmark_return for result in results] == [0.05, 0.15, 0.25]


def test_insufficient_history_does_not_mark_horizon_complete() -> None:
    evaluator, store = _evaluator(_market_csv(history_days=14))

    results = asyncio.run(evaluator.evaluate_single(_record()))

    assert results == []
    assert store.marked == []


def test_failed_explanation_falls_back_to_objective_result() -> None:
    evaluator, store = _evaluator(_market_csv())
    evaluator._generate_lesson = AsyncMock(
        side_effect=[None, "评估结果: 可解释\n经验教训: 可复用", "评估结果: 可解释\n经验教训: 可复用"],
    )

    results = asyncio.run(evaluator.evaluate_single(_record()))

    assert [result.horizon_days for result in results] == [1, 3, 5]
    assert results[0].evaluation == "客观状态: correct；是否成交: 否；基准收益率: +5.00%；策略收益率: -"
    assert results[0].lesson == ""
    assert [marked[2] for marked in store.marked] == [1, 3, 5]


def test_horizon_exception_does_not_block_later_horizons() -> None:
    evaluator, store = _evaluator(_market_csv())
    evaluator._generate_lesson = AsyncMock(
        side_effect=[RuntimeError("horizon failed"), "评估结果: 可解释\n经验教训: 可复用", "评估结果: 可解释\n经验教训: 可复用"],
    )

    results = asyncio.run(evaluator.evaluate_single(_record()))

    assert [result.horizon_days for result in results] == [3, 5]
    assert [marked[2] for marked in store.marked] == [3, 5]


def test_dry_run_never_marks_completed_horizons() -> None:
    evaluator, store = _evaluator(_market_csv(), dry_run=True)
    lessons_memory = SimpleNamespace(add_reflection_lesson=AsyncMock())
    evaluator._memory_manager = SimpleNamespace(lessons_memory=lessons_memory)

    results = asyncio.run(evaluator.evaluate_single(_record()))

    assert [result.horizon_days for result in results] == [1, 3, 5]
    assert store.marked == []
    lessons_memory.add_reflection_lesson.assert_not_awaited()


def test_batch_flattens_results_and_one_ticker_failure_does_not_block_another() -> None:
    evaluator, store = _evaluator(_market_csv())
    store.records = [_record("AAPL"), _record("MSFT")]
    expected = EvaluationResult("MSFT", "ok", "lesson", horizon_days=1)
    evaluator.evaluate_single = AsyncMock(side_effect=[RuntimeError("bad data"), [expected]])

    results = asyncio.run(evaluator.run_batch_evaluation())

    assert results == [expected]


def test_lesson_type_is_derived_from_objective_status() -> None:
    evaluator, _store = _evaluator(_market_csv())
    lessons_memory = SimpleNamespace(add_reflection_lesson=AsyncMock())
    evaluator._memory_manager = SimpleNamespace(lessons_memory=lessons_memory)

    asyncio.run(evaluator.evaluate_single(_record()))

    lesson_types = [call.kwargs["lesson_type"] for call in lessons_memory.add_reflection_lesson.await_args_list]
    assert lesson_types == ["success", "success", "success"]


def test_all_objective_statuses_map_to_the_expected_lesson_type() -> None:
    assert _lesson_type_for_status(AssessmentStatus.CORRECT) == "success"
    assert _lesson_type_for_status(AssessmentStatus.INCORRECT) == "failure"
    assert _lesson_type_for_status(AssessmentStatus.INCONCLUSIVE) == "partial"
    assert _lesson_type_for_status(AssessmentStatus.NOT_FILLED) == "partial"
