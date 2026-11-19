"""Offline tests for progressive, deterministic evaluation and run identity."""

from datetime import date
from unittest.mock import Mock, patch

import pytest

from tests.support.evaluation import (
    AFTER_CLOSE_RELEASE,
    ANALYSIS_DATE,
    EVALUATED_AT,
    PREMARKET_RELEASE,
    FakeMarketData,
    completed_run,
    flat_future,
    market_bars,
)
from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator, build_evaluation_record, current_records, trade_record
from tradingscope.agents.evaluation.models import SampleClass, ShockClass
from tradingscope.agents.evaluation.oss_store import LocalEvaluationStore


def _evaluator(tmp_path, *, days=5, dry_run=False, run=None, fetch_run=None):
    run = run or completed_run(direction="bullish", action="buy")
    data = FakeMarketData(market_bars(flat_future([101 + index for index in range(days)])))
    store = LocalEvaluationStore(tmp_path)
    evaluator = AnalysisEvaluator(
        store,
        market_data=data,
        fetch_run=fetch_run or (lambda *_args: run),
        dry_run=dry_run,
        clock=lambda: EVALUATED_AT,
    )
    return evaluator, store, data


def test_only_mature_pending_horizons_are_assessed_and_written(tmp_path):
    evaluator, store, _ = _evaluator(tmp_path, days=2)
    records = evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat())
    assert [record.horizon for record in records] == [1]
    assert [record.horizon for record in store.list_records()] == [1]


def test_all_stages_have_direction_results_and_only_trader_pm_have_simulations(tmp_path):
    evaluator, _, _ = _evaluator(tmp_path)
    records = evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat())
    assert [record.horizon for record in records] == [1]
    assert all(len(record.stages) == 7 for record in records)
    assert {stage.stage for stage in records[0].stages if stage.trade is not None} == {"trader", "portfolio_manager"}


def test_evaluation_is_deterministic_given_inputs_and_timestamp(tmp_path):
    evaluator, _, data = _evaluator(tmp_path)
    run = completed_run(direction="bullish", action="buy")
    args = (run, data.bars["AAPL"], "XLK", data.bars["XLK"], None, 1)
    first = build_evaluation_record(*args, evaluated_at=EVALUATED_AT)
    second = build_evaluation_record(*args, evaluated_at=EVALUATED_AT)
    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.parametrize("horizon", [3, 5])
def test_evaluation_record_rejects_multi_session_horizons(horizon):
    with pytest.raises(ValueError, match="allowed horizons are 1"):
        build_evaluation_record(
            completed_run(),
            market_bars(flat_future([101.0] * 5)),
            "XLK",
            market_bars(flat_future([100.0] * 5)),
            [],
            horizon,
            evaluated_at=EVALUATED_AT,
        )


def test_next_session_record_does_not_change_with_later_prices_or_earnings():
    from datetime import datetime

    run = completed_run(
        direction="bullish",
        action="buy",
        portfolio_updates={
            "trade_intent": "open_long",
            "time_stop_days": 5,
            "price_plan": {"execution_price": 100.0, "target_price": 110.0, "stop_loss": 95.0},
        },
    )
    first = build_evaluation_record(run, market_bars(flat_future([103.0])), "XLK", market_bars(flat_future([100.0])), [], 1, evaluated_at=EVALUATED_AT)
    extended = build_evaluation_record(
        run,
        market_bars(flat_future([103.0, 120.0, 80.0, 110.0, 90.0])),
        "XLK",
        market_bars(flat_future([100.0, 110.0, 100.0, 95.0, 100.0])),
        [datetime(2026, 7, 17, 8, tzinfo=PREMARKET_RELEASE.tzinfo)],
        1,
        evaluated_at=EVALUATED_AT,
    )

    assert extended == first
    assert first.horizon_end_session == ANALYSIS_DATE
    assert first.market.underlying_return == pytest.approx(0.03)
    assert first.stage("portfolio_manager").trade.exit_reason == "horizon"
    assert first.source_outputs["portfolio_manager"]["time_stop_days"] == 5


def test_already_written_horizons_do_not_refetch_market_data(tmp_path):
    evaluator, store, data = _evaluator(tmp_path)
    assert len(evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat())) == 1
    data.calls.clear()
    assert evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat()) == []
    assert data.calls == []
    assert len(store.list_records()) == 1


def test_market_window_is_prepared_once_per_horizon(tmp_path):
    from tradingscope.agents.evaluation import market_outcome

    evaluator, _, _ = _evaluator(tmp_path)
    with patch.object(market_outcome, "_average_true_range", wraps=market_outcome._average_true_range) as calculate_atr:
        records = evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat())
    assert [record.horizon for record in records] == [1]
    assert calculate_atr.call_count == 1


def test_market_data_is_fetched_once_per_run_not_once_per_horizon(tmp_path):
    evaluator, _, data = _evaluator(tmp_path)
    evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat())
    assert [call[1] for call in data.calls] == ["AAPL", "XLK"]
    assert data.calls[0] == ("bars", "AAPL", "2026-06-09", "2026-07-28")


def test_new_run_id_reassesses_same_date_without_modifying_old_records(tmp_path):
    current = completed_run(run_id="run-1")
    evaluator, store, _ = _evaluator(tmp_path, fetch_run=lambda *_args: current)
    first = evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat())
    current = completed_run(run_id="run-2")
    second = evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat())
    assert len(first) == len(second) == 1
    assert len(store.list_records()) == 2
    assert {record.run_id for record in current_records(store.list_records(), lambda *_args: current.manifest)} == {"run-2"}


def test_current_utc_session_cannot_be_persisted_as_matured(tmp_path):
    from datetime import datetime, timezone

    evaluator, store, _ = _evaluator(tmp_path)
    evaluator._clock = lambda: datetime(2026, 7, 14, 12, tzinfo=timezone.utc)
    assert evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat()) == []
    assert store.list_records() == []


def test_current_records_excludes_archived_multi_session_evaluations():
    from tests.support.evaluation import evaluation_record

    record = evaluation_record()
    archived = [record.model_copy(update={"horizon": horizon, "horizon_end_session": date(2026, 7, 20)}) for horizon in (3, 5)]
    manifest = completed_run().manifest

    assert current_records([record, *archived], lambda *_args: manifest) == [record]


def test_superseded_run_is_excluded_even_if_no_new_evaluation_exists():
    record = build_evaluation_record(
        completed_run(),
        market_bars(flat_future([101.0] * 5)),
        "XLK",
        market_bars(flat_future([100.0] * 5)),
        [],
        1,
        evaluated_at=EVALUATED_AT,
    )
    new_manifest = completed_run(run_id="replacement").manifest
    assert current_records([record], lambda *_args: new_manifest) == []


def test_dry_run_never_writes_records(tmp_path):
    evaluator, store, _ = _evaluator(tmp_path, dry_run=True)
    assert len(evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat())) == 1
    assert store.list_records() == []


def test_insufficient_history_or_missing_benchmark_session_leaves_horizon_pending(tmp_path):
    evaluator, store, data = _evaluator(tmp_path)
    data.bars["AAPL"] = market_bars(flat_future([101.0] * 5), history=14)
    assert evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat()) == []
    data.bars["AAPL"] = market_bars(flat_future([101.0] * 5))
    data.bars["XLK"] = []
    assert evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat()) == []
    assert store.list_records() == []


def test_failed_ticker_does_not_block_other_candidates(tmp_path):
    evaluator, _, _ = _evaluator(tmp_path)
    run = completed_run()
    evaluator._fetch_run = Mock(side_effect=[ValueError("bad data"), run])
    records = evaluator.run(["MSFT", "AAPL"], analysis_date=ANALYSIS_DATE.isoformat())
    assert len(records) == 1


def test_trade_outcome_preserves_reduce_long_scope_and_residual_exit(tmp_path):
    run = completed_run(
        direction="bullish",
        action="sell",
        portfolio_updates={
            "trade_intent": "reduce_long",
            "position_advice": "light",
            "time_stop_days": 3,
            "price_plan": {"execution_price": 100.0, "target_price": 110.0, "stop_loss": 95.0},
        },
        stages=("portfolio_manager",),
    )
    evaluator, _, _ = _evaluator(tmp_path, run=run)
    records = evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat())
    trade = records[-1].stage("portfolio_manager").trade
    assert trade.strategy_return_basis == "exit_relative_performance"
    assert trade.residual_exit_reason == "horizon"
    assert trade.residual_exit_date == "2026-07-14"
    assert trade.residual_exit_price == 101.0
    assert trade.limitations
    record = trade_record(run.outputs["portfolio_manager"])
    assert record.trade_intent == "reduce_long"


def test_post_close_forecast_does_not_score_an_already_observed_session():
    run = completed_run()
    run.manifest.last_completed_session = ANALYSIS_DATE
    with pytest.raises(ValueError, match="already-observed"):
        build_evaluation_record(
            run,
            market_bars(flat_future([105.0] * 5)),
            "XLK",
            market_bars(flat_future([100.0] * 5)),
            [],
            1,
            evaluated_at=EVALUATED_AT,
        )


@pytest.mark.parametrize(("hour", "minute"), [(13, 30), (15, 0)])
def test_forecast_completed_intraday_cannot_score_the_full_daily_bar(hour, minute):
    from datetime import datetime, timezone

    run = completed_run()
    run.manifest.forecast_completed_at = datetime(2026, 7, 14, hour, minute, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="after its first session opened"):
        build_evaluation_record(
            run,
            market_bars(flat_future([105.0] * 5)),
            "XLK",
            market_bars(flat_future([100.0] * 5)),
            [],
            1,
            evaluated_at=EVALUATED_AT,
        )


def test_evening_earnings_does_not_excuse_a_morning_shock():
    record = build_evaluation_record(
        completed_run(),
        market_bars(flat_future([120.0] * 5)),
        "XLK",
        market_bars(flat_future([100.0] * 5)),
        [AFTER_CLOSE_RELEASE],
        1,
        evaluated_at=EVALUATED_AT,
    )
    assert record.market.shock_sessions[0].classification is ShockClass.UNSCHEDULED
    assert record.market.sample_class is SampleClass.SHOCK_EXCLUDED


def test_premarket_earnings_does_not_excuse_the_following_session():
    record = build_evaluation_record(
        completed_run(analysis_date=date(2026, 7, 15)),
        market_bars(flat_future([100.0, 120.0, 120.0])),
        "XLK",
        market_bars(flat_future([100.0] * 3)),
        [PREMARKET_RELEASE],
        1,
        evaluated_at=EVALUATED_AT,
    )
    assert record.market.shock_sessions[0].classification is ShockClass.UNSCHEDULED


def test_period_return_drops_market_move_from_direction_skill(tmp_path):
    run = completed_run(direction="bullish", action="buy")
    ticker = market_bars(flat_future([105.0] * 5))
    benchmark = market_bars(flat_future([108.0] * 5))
    record = build_evaluation_record(run, ticker, "XLK", benchmark, [], 1, evaluated_at=EVALUATED_AT)
    pm = record.stage("portfolio_manager")
    assert pm.direction_status == "correct"
    assert pm.excess_direction_status == "incorrect"
    assert record.market.excess_return == pytest.approx(-0.03)


@pytest.mark.parametrize(
    ("earnings", "classification", "sample"),
    [
        (None, ShockClass.UNKNOWN, SampleClass.SHOCK_EXCLUDED),
        ([], ShockClass.UNSCHEDULED, SampleClass.SHOCK_EXCLUDED),
        ([PREMARKET_RELEASE], ShockClass.SCHEDULED, SampleClass.CLEAN),
    ],
)
def test_shock_classification_controls_direction_sample(earnings, classification, sample):
    ticker = market_bars(flat_future([120.0] * 5))
    benchmark = market_bars(flat_future([100.0] * 5))
    record = build_evaluation_record(completed_run(), ticker, "XLK", benchmark, earnings, 1, evaluated_at=EVALUATED_AT)
    assert record.market.shock_sessions[0].classification is classification
    assert record.market.sample_class is sample


def test_after_close_earnings_shock_is_scheduled_on_the_next_session():
    ticker = market_bars(flat_future([100.0, 120.0, 120.0]))
    benchmark = market_bars(flat_future([100.0] * 3))
    record = build_evaluation_record(
        completed_run(analysis_date=date(2026, 7, 15)), ticker, "XLK", benchmark, [AFTER_CLOSE_RELEASE], 1, evaluated_at=EVALUATED_AT
    )
    assert record.market.shock_sessions[0].session_date == date(2026, 7, 15)
    assert record.market.shock_sessions[0].classification is ShockClass.SCHEDULED
    assert record.market.sample_class is SampleClass.CLEAN


def test_market_wide_jump_is_not_an_individual_shock():
    bars = market_bars([(120.0, 121.0, 119.0, 120.0)] * 5)
    record = build_evaluation_record(completed_run(), bars, "XLK", bars, [], 1, evaluated_at=EVALUATED_AT)
    assert record.market.shock_sessions == []
    assert record.market.excess_return == 0


def test_unfilled_plan_is_a_trade_outcome_not_a_direction_failure(tmp_path):
    run = completed_run(
        direction="bullish",
        action="buy",
        portfolio_updates={"trade_intent": "open_long", "price_plan": {"execution_price": 90.0, "target_price": 110.0, "stop_loss": 85.0}},
    )
    evaluator, _, data = _evaluator(tmp_path, run=run)
    data.bars["AAPL"] = market_bars(flat_future([105.0] * 5))
    record = evaluator.run(["AAPL"], analysis_date=ANALYSIS_DATE.isoformat())[-1]
    pm = record.stage("portfolio_manager")
    assert pm.trade.status == "not_filled"
    assert pm.trade.simulated
    assert pm.excess_direction_status == "correct"
