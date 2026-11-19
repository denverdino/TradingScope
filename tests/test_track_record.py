from datetime import date

import pytest

from tests.support.evaluation import EVALUATED_AT, PREMARKET_RELEASE, completed_run, flat_future, market_bars
from tradingscope.agents.evaluation.evaluator import build_evaluation_record
from tradingscope.agents.evaluation.track_record import build_track_record


def _record(close=105.0, *, earnings=(), run=None):
    return build_evaluation_record(
        run or completed_run(direction="bullish", action="buy"),
        market_bars(flat_future([close] * 5)),
        "XLK",
        market_bars(flat_future([100.0] * 5)),
        list(earnings),
        1,
        evaluated_at=EVALUATED_AT,
    )


def _track(records, **kwargs):
    return build_track_record(records, min_samples=10, confidence_buckets=(0.5, 0.7), **kwargs)


def _cell(track, stage="portfolio_manager", scope="ALL"):
    return next(cell for cell in track.cells if cell.stage == stage and cell.scope == scope)


def test_small_sample_suppresses_rates():
    cell = _cell(_track([_record()] * 9))
    assert cell.direction.n == 9
    assert cell.direction.hit_rate is None
    assert cell.direction.inconclusive_share is None
    assert cell.calibration[-1].skill.hit_rate is None


def test_hit_rate_excludes_inconclusive_but_reports_its_share():
    cell = _cell(_track([_record()] * 6 + [_record(95.0)] * 3 + [_record(100.5)]))
    assert cell.direction.n == 10
    assert cell.direction.hit_rate == 6 / 9
    assert cell.direction.inconclusive_share == 0.1
    assert cell.calibration[-1].skill.n == 10


def test_future_horizons_and_pre_reset_dates_are_excluded():
    assert _track([_record()], covered_through=date(2026, 7, 13)).cells == []
    assert _track([_record()], start_date=date(2026, 7, 15)).cells == []
    assert _cell(_track([_record()], covered_through=date(2026, 7, 14))).direction.n == 1


def test_unscheduled_shocks_are_excluded_from_direction_but_not_risk():
    run = completed_run(
        direction="bullish",
        action="buy",
        portfolio_updates={"trade_intent": "open_long", "price_plan": {"execution_price": 100.0, "stop_loss": 95.0, "target_price": 110.0}},
    )
    shocked = _record(120.0, run=run)
    cell = _cell(_track([shocked] * 10))
    assert cell.direction.n == 0
    assert cell.risk.n == 10
    assert cell.risk.fill_rate == 1.0
    assert cell.risk.mean_strategy_return is not None


def test_scheduled_shock_counts_in_direction_skill():
    cell = _cell(_track([_record(120.0, earnings=[PREMARKET_RELEASE])] * 10))
    assert cell.direction.n == 10
    assert cell.direction.hit_rate == 1.0


def test_incompatible_trade_intents_are_not_pooled_into_one_return():
    opened = completed_run(
        direction="bullish",
        action="buy",
        portfolio_updates={"trade_intent": "open_long", "price_plan": {"execution_price": 100.0, "stop_loss": 95.0, "target_price": 110.0}},
    )
    reduced = completed_run(
        direction="bullish",
        action="sell",
        stages=("portfolio_manager",),
        portfolio_updates={
            "trade_intent": "reduce_long",
            "position_advice": "light",
            "time_stop_days": 1,
            "price_plan": {"execution_price": 100.0, "stop_loss": 95.0, "target_price": 110.0},
        },
    )
    cell = _cell(_track([_record(run=opened)] * 10 + [_record(90.0, run=reduced)] * 10))
    assert cell.risk.mean_strategy_return is None
    assert {item.trade_intent for item in cell.risk.return_performance} == {"open_long", "reduce_long"}
    assert cell.risk.residual_stop_losses == 10
    assert cell.risk.residual_stop_rate == 1.0


@pytest.mark.parametrize("trade_intent", ["open_long", "reduce_long"])
def test_reports_distinguish_closing_valuation_from_exit(trade_intent):
    from tradingscope.agents.evaluation.report import record_headline, render_records_html

    run = completed_run(
        direction="bullish",
        action="buy" if trade_intent == "open_long" else "sell",
        stages=("portfolio_manager",),
        portfolio_updates={
            "trade_intent": trade_intent,
            "position_advice": "light",
            "time_stop_days": 5,
            "price_plan": {"execution_price": 100.0, "stop_loss": 95.0, "target_price": 110.0},
        },
    )
    record = _record(103.0, run=run)
    trade = record.stage("portfolio_manager").trade
    if trade_intent == "open_long":
        assert trade.exit_reason == "horizon"
    else:
        assert trade.exit_reason == "execution"
        assert trade.residual_exit_reason == "horizon"

    for rendered in (record_headline(record), render_records_html([record])):
        assert "评估交易日：2026-07-14" in rendered
        assert "收盘估值（未退出）" in rendered
        assert "退出=horizon" not in rendered


def test_reports_include_calibration_and_residual_scope():
    from tradingscope.agents.evaluation.report import render_records_html, render_track_record_html, render_track_record_text

    reduced = completed_run(
        direction="bullish",
        action="sell",
        stages=("portfolio_manager",),
        portfolio_updates={
            "trade_intent": "reduce_long",
            "position_advice": "light",
            "time_stop_days": 1,
            "price_plan": {"execution_price": 100.0, "stop_loss": 95.0, "target_price": 110.0},
        },
    )
    record = _record(90.0, run=reduced)
    track = _track([record] * 10)
    html = render_track_record_html(track)
    assert "置信度校准" in html
    assert "70%" in html
    assert "reduce_long/exit_relative_performance" in html
    assert "剩余仓位止损率" in html
    assert "置信度" in render_track_record_text(track)
    assert "cost basis" in render_records_html([record])


def test_archived_multi_session_records_do_not_enter_any_statistics():
    record = _record()
    archived = [record.model_copy(update={"horizon": horizon, "horizon_end_session": date(2026, 7, 20)}) for horizon in (3, 5)]

    expected = _track([record])
    actual = _track([record, *archived])

    assert actual.model_dump() == expected.model_dump()
    assert _track(archived).cells == []
    assert _track(archived).attribution == []


def test_ticker_and_global_groups_are_separate():
    record = _record()
    msft = record.model_copy(update={"ticker": "MSFT"})
    track = _track([record, msft])
    assert _cell(track).direction.n == 2
    assert _cell(track, scope="AAPL").direction.n == 1
    assert _cell(track, scope="MSFT").direction.n == 1


def test_attribution_compares_agreed_and_disagreed_portfolio_calls():
    run = completed_run(direction="bullish", action="buy", analyst_directions={"news_analyst": "bullish"})
    record = _record(run=run)
    track = _track([record] * 10)
    news = next(item for item in track.attribution if item.scope == "ALL" and item.analyst_stage == "news_analyst")
    market = next(item for item in track.attribution if item.scope == "ALL" and item.analyst_stage == "market_analyst")
    assert news.agreed.hit_rate == 1.0
    assert news.disagreed.n == 0
    assert market.agreed.n == 0
    assert market.disagreed.hit_rate == 1.0
