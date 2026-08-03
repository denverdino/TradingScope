from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from tradingscope.agents.evaluation.market_outcome import (
    AssessmentStatus,
    MarketBar,
    assess_trade,
    parse_market_bars,
)
from tradingscope.agents.evaluation.models import AnalysisRecord


def _record(**overrides: object) -> AnalysisRecord:
    values: dict[str, object] = {
        "ticker": "AAPL",
        "trade_date": "2026-01-15",
        "direction": "bullish",
        "action": "buy",
        "confidence": 0.8,
    }
    values.update(overrides)
    return AnalysisRecord(**values)  # type: ignore[arg-type]


def _market_csv() -> str:
    rows = ["Timestamp,OPEN,High,low,CLOSE"]
    rows.append("2025-12-31,100,101,99,100")
    for day in range(1, 14):
        rows.append(f"2026-01-{day:02d},100,101,99,100")
    # Gap from the previous 100 close makes this true range 11, not 2.
    rows.append("2026-01-14,110,111,109,110")
    # Analysis-day and future ranges must never enter the ATR calculation.
    rows.extend(
        [
            "2026-01-15,110,1000,1,113",
            "2026-01-16,113,1000,1,114",
            "2026-01-17,114,115,113,115",
            "2026-01-18,115,116,114,116",
            "2026-01-19,116,117,115,117",
            "2026-01-20,117,118,116,118",
        ]
    )
    return "\n".join(reversed(rows[1:])) + "\n" + rows[0]


def _constant_atr_bars(future_close: float, future_count: int = 5) -> list[MarketBar]:
    start = date(2025, 12, 31)
    bars = [MarketBar(start + timedelta(days=index), 100.0, 101.0, 99.0, 100.0) for index in range(15)]
    bars.extend(
        MarketBar(
            date(2026, 1, 15) + timedelta(days=index),
            100.0,
            max(101.0, future_close),
            min(99.0, future_close),
            future_close,
        )
        for index in range(future_count)
    )
    return bars


def _execution_bars(future_ohlc: list[tuple[float, float, float, float]]) -> list[MarketBar]:
    bars = _constant_atr_bars(100.0, future_count=0)
    bars.extend(MarketBar(date(2026, 1, 15) + timedelta(days=index), *ohlc) for index, ohlc in enumerate(future_ohlc))
    return bars


def test_parse_market_bars_accepts_case_insensitive_ohlc_and_sorts_dates() -> None:
    csv_lines = _market_csv().splitlines()
    csv_text = "\n".join([csv_lines[-1], *csv_lines[:-1]])

    bars = parse_market_bars(csv_text)

    assert len(bars) == 21
    assert bars[0].trade_date.isoformat() == "2025-12-31"
    assert bars[-1].trade_date.isoformat() == "2026-01-20"
    assert bars[14].open == 110.0
    assert bars[14].high == 111.0
    assert bars[14].low == 109.0
    assert bars[14].close == 110.0


@pytest.mark.parametrize("bad_price", ["0", "-1", "nan", "inf"])
def test_parse_market_bars_rejects_non_positive_or_non_finite_prices(bad_price: str) -> None:
    csv_text = f"timestamp,open,high,low,close\n2026-01-01,100,101,99,{bad_price}"

    with pytest.raises(ValueError, match="price"):
        parse_market_bars(csv_text)


def test_parse_market_bars_rejects_duplicate_dates() -> None:
    csv_text = "\n".join(
        [
            "timestamp,open,high,low,close",
            "2026-01-01,100,101,99,100",
            "2026-01-01,101,102,100,101",
        ]
    )

    with pytest.raises(ValueError, match="duplicate"):
        parse_market_bars(csv_text)


def test_parse_market_bars_skips_vendor_row_when_all_ohlc_values_are_missing() -> None:
    csv_text = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits",
            "2026-07-23,115.76,116.41,113.1,114.06,8500200,0.0,0.0",
            "2026-07-24,,,,,6586066,0.0,0.0",
        ]
    )

    bars = parse_market_bars(csv_text)

    assert [bar.trade_date.isoformat() for bar in bars] == ["2026-07-23"]


def test_parse_market_bars_rejects_partially_missing_ohlc_values() -> None:
    csv_text = "\n".join(
        [
            "Date,Open,High,Low,Close,Volume",
            "2026-07-23,115.76,116.41,,114.06,8500200",
        ]
    )

    with pytest.raises(ValueError, match="invalid market bar"):
        parse_market_bars(csv_text)


def test_atr_uses_true_range_from_only_fourteen_pre_analysis_bars() -> None:
    csv_lines = _market_csv().splitlines()
    bars = parse_market_bars("\n".join([csv_lines[-1], *csv_lines[:-1]]))

    assessment = assess_trade(_record(), bars, horizon_days=1)

    assert assessment is not None
    assert assessment.atr_threshold == pytest.approx((11.0 + 13 * 2.0) / 14 / 110.0)
    assert assessment.benchmark_return == pytest.approx(3.0 / 110.0)
    assert assessment.status is AssessmentStatus.CORRECT


def test_preopen_record_uses_previous_close_and_current_session_as_horizon_one() -> None:
    assessment = assess_trade(
        _record(trade_date="2026-01-15"),
        _constant_atr_bars(103.0, future_count=1),
        horizon_days=1,
    )

    assert assessment is not None
    assert assessment.benchmark_return == pytest.approx(0.03)


def test_weekend_record_uses_friday_close_and_monday_as_horizon_one() -> None:
    friday = date(2026, 1, 16)
    history = [MarketBar(friday - timedelta(days=14 - index), 100.0, 101.0, 99.0, 100.0) for index in range(15)]
    monday = MarketBar(date(2026, 1, 19), 100.0, 104.0, 99.0, 103.0)

    assessment = assess_trade(
        _record(trade_date="2026-01-17"),
        [*history, monday],
        horizon_days=1,
    )

    assert assessment is not None
    assert assessment.benchmark_return == pytest.approx(0.03)


def test_only_horizons_with_enough_evaluation_sessions_are_available() -> None:
    bars = _constant_atr_bars(103.0, future_count=3)

    assert assess_trade(_record(), bars, horizon_days=1) is not None
    assert assess_trade(_record(), bars, horizon_days=3) is not None
    assert assess_trade(_record(), bars, horizon_days=5) is None


def test_atr_threshold_scales_by_square_root_of_horizon() -> None:
    csv_lines = _market_csv().splitlines()
    bars = parse_market_bars("\n".join([csv_lines[-1], *csv_lines[:-1]]))

    one_day = assess_trade(_record(), bars, horizon_days=1)
    three_day = assess_trade(_record(), bars, horizon_days=3)

    assert one_day is not None
    assert three_day is not None
    assert three_day.atr_threshold == pytest.approx(one_day.atr_threshold * math.sqrt(3))


def test_atr_uses_previous_close_before_the_fourteen_range_window() -> None:
    bars = [MarketBar(date(2025, 12, 31), 100.0, 101.0, 99.0, 100.0)]
    bars.append(MarketBar(date(2026, 1, 1), 110.0, 111.0, 109.0, 110.0))
    bars.extend(MarketBar(date(2026, 1, 1) + timedelta(days=index), 110.0, 111.0, 109.0, 110.0) for index in range(1, 14))
    bars.extend(
        [
            MarketBar(date(2026, 1, 15), 110.0, 111.0, 109.0, 110.0),
            MarketBar(date(2026, 1, 16), 110.0, 114.0, 109.0, 113.0),
        ]
    )

    assessment = assess_trade(_record(), bars, horizon_days=1)

    assert assessment is not None
    assert assessment.atr_threshold == pytest.approx((11.0 + 13 * 2.0) / 14 / 110.0)


def test_atr_assessment_requires_previous_close_plus_fourteen_ranges() -> None:
    bars = [MarketBar(date(2026, 1, 1) + timedelta(days=index), 100.0, 101.0, 99.0, 100.0) for index in range(14)]
    bars.extend(
        [
            MarketBar(date(2026, 1, 15), 100.0, 101.0, 99.0, 100.0),
            MarketBar(date(2026, 1, 16), 100.0, 104.0, 99.0, 103.0),
        ]
    )

    assert assess_trade(_record(), bars, horizon_days=1) is None


@pytest.mark.parametrize("invalid_horizon", [True, False, 1.0, 3.0, 5.0, 0, 2, 4, -1])
def test_assess_trade_rejects_non_exact_supported_horizons(invalid_horizon: object) -> None:
    with pytest.raises(ValueError, match="allowed horizons are 1, 3, 5"):
        assess_trade(
            _record(),
            _constant_atr_bars(103.0),
            horizon_days=invalid_horizon,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("direction", "future_close", "expected"),
    [
        ("bullish", 103.0, AssessmentStatus.CORRECT),
        ("bullish", 97.0, AssessmentStatus.INCORRECT),
        ("bearish", 97.0, AssessmentStatus.CORRECT),
        ("bearish", 103.0, AssessmentStatus.INCORRECT),
    ],
)
def test_direction_requires_movement_beyond_atr_threshold(
    direction: str,
    future_close: float,
    expected: AssessmentStatus,
) -> None:
    assessment = assess_trade(
        _record(direction=direction),
        _constant_atr_bars(future_close),
        horizon_days=1,
    )

    assert assessment is not None
    assert assessment.status is expected


@pytest.mark.parametrize(
    ("direction", "future_close"),
    [
        ("bullish", 99.0),
        ("bullish", 98.0),
        ("bearish", 101.0),
        ("bearish", 102.0),
    ],
)
def test_direction_inside_or_at_atr_boundary_is_inconclusive(
    direction: str,
    future_close: float,
) -> None:
    assessment = assess_trade(
        _record(direction=direction),
        _constant_atr_bars(future_close),
        horizon_days=1,
    )

    assert assessment is not None
    assert assessment.status is AssessmentStatus.INCONCLUSIVE


@pytest.mark.parametrize(
    ("future_close", "expected"),
    [
        (98.0, AssessmentStatus.CORRECT),
        (102.0, AssessmentStatus.CORRECT),
        (97.0, AssessmentStatus.INCORRECT),
        (103.0, AssessmentStatus.INCORRECT),
    ],
)
def test_neutral_direction_includes_both_atr_boundaries(
    future_close: float,
    expected: AssessmentStatus,
) -> None:
    assessment = assess_trade(
        _record(direction="neutral"),
        _constant_atr_bars(future_close),
        horizon_days=1,
    )

    assert assessment is not None
    assert assessment.status is expected


@pytest.mark.parametrize(
    "bars",
    [
        _constant_atr_bars(103.0)[1:],
        _constant_atr_bars(103.0)[:14],
        _constant_atr_bars(103.0, future_count=2),
    ],
)
def test_direction_assessment_requires_base_history_and_requested_horizon(
    bars: list[MarketBar],
) -> None:
    assert assess_trade(_record(), bars, horizon_days=3) is None


@pytest.mark.parametrize(
    ("record", "bars", "expected_entry"),
    [
        (
            _record(
                direction="bullish",
                trade_intent="open_long",
                entry_price=100.0,
                entry_price_low=99.0,
                entry_price_high=101.0,
                target_price=110.0,
                stop_loss=90.0,
                time_stop_days=3,
            ),
            _execution_bars(
                [
                    (103.0, 104.0, 102.0, 103.0),
                    (102.0, 102.5, 100.5, 101.0),
                    (101.0, 102.0, 100.0, 101.0),
                ]
            ),
            100.5,
        ),
        (
            _record(
                direction="bearish",
                trade_intent="open_short",
                entry_price=101.0,
                entry_price_low=100.0,
                entry_price_high=102.0,
                target_price=90.0,
                stop_loss=110.0,
                time_stop_days=3,
            ),
            _execution_bars(
                [
                    (97.0, 99.0, 96.0, 98.0),
                    (99.0, 100.5, 98.0, 100.0),
                    (100.0, 101.0, 99.0, 100.0),
                ]
            ),
            100.5,
        ),
    ],
)
def test_open_trade_uses_first_fill_and_clamps_representative_price(
    record: AnalysisRecord,
    bars: list[MarketBar],
    expected_entry: float,
) -> None:
    assessment = assess_trade(record, bars, horizon_days=3)

    assert assessment is not None
    assert assessment.entry_triggered is True
    assert assessment.entry_price == expected_entry


@pytest.mark.parametrize("trade_intent", ["open_long", "open_short"])
def test_open_trade_range_never_touched_is_not_filled(trade_intent: str) -> None:
    is_long = trade_intent == "open_long"
    assessment = assess_trade(
        _record(
            direction="bullish" if is_long else "bearish",
            trade_intent=trade_intent,
            entry_price=90.5,
            entry_price_low=90.0,
            entry_price_high=91.0,
            target_price=110.0 if is_long else 80.0,
            stop_loss=80.0 if is_long else 110.0,
            time_stop_days=3,
        ),
        _execution_bars([(100.0, 102.0, 98.0, 100.0)] * 3),
        horizon_days=3,
    )

    assert assessment is not None
    assert assessment.status is AssessmentStatus.NOT_FILLED
    assert assessment.entry_triggered is False
    assert assessment.entry_price is None
    assert assessment.exit_reason == "not_filled"
    assert assessment.strategy_return is None


def test_open_trade_does_not_fill_after_time_stop_deadline() -> None:
    assessment = assess_trade(
        _record(
            trade_intent="open_long",
            entry_price=100.0,
            entry_price_low=99.0,
            entry_price_high=101.0,
            target_price=120.0,
            stop_loss=80.0,
            time_stop_days=2,
        ),
        _execution_bars(
            [
                (103.0, 104.0, 102.0, 103.0),
                (102.0, 103.0, 101.5, 102.0),
                (100.0, 101.0, 99.0, 100.0),
            ]
        ),
        horizon_days=3,
    )

    assert assessment is not None
    assert assessment.status is AssessmentStatus.NOT_FILLED
    assert assessment.entry_triggered is False
    assert assessment.entry_price is None
    assert assessment.exit_reason == "not_filled"
    assert assessment.strategy_return is None


@pytest.mark.parametrize(
    ("high", "low", "expected_reason", "expected_return", "expected_status"),
    [
        (103.0, 98.0, "target", 0.03, AssessmentStatus.CORRECT),
        (102.0, 97.0, "stop_loss", -0.03, AssessmentStatus.INCORRECT),
        (103.0, 97.0, "stop_loss", -0.03, AssessmentStatus.INCORRECT),
    ],
)
def test_open_long_target_stop_and_same_day_stop_precedence(
    high: float,
    low: float,
    expected_reason: str,
    expected_return: float,
    expected_status: AssessmentStatus,
) -> None:
    assessment = assess_trade(
        _record(
            trade_intent="open_long",
            entry_price=100.0,
            entry_price_low=99.0,
            entry_price_high=101.0,
            target_price=103.0,
            stop_loss=97.0,
            time_stop_days=1,
        ),
        _execution_bars([(100.0, high, low, 100.0)]),
        horizon_days=1,
    )

    assert assessment is not None
    assert assessment.exit_reason == expected_reason
    assert assessment.strategy_return == pytest.approx(expected_return)
    assert assessment.status is expected_status


@pytest.mark.parametrize(
    ("high", "low", "expected_reason", "expected_return", "expected_status"),
    [
        (102.0, 97.0, "target", 0.03, AssessmentStatus.CORRECT),
        (103.0, 98.0, "stop_loss", -0.03, AssessmentStatus.INCORRECT),
        (103.0, 97.0, "stop_loss", -0.03, AssessmentStatus.INCORRECT),
    ],
)
def test_open_short_target_stop_and_same_day_stop_precedence(
    high: float,
    low: float,
    expected_reason: str,
    expected_return: float,
    expected_status: AssessmentStatus,
) -> None:
    assessment = assess_trade(
        _record(
            direction="bearish",
            trade_intent="open_short",
            entry_price=100.0,
            entry_price_low=99.0,
            entry_price_high=101.0,
            target_price=97.0,
            stop_loss=103.0,
            time_stop_days=1,
        ),
        _execution_bars([(100.0, high, low, 100.0)]),
        horizon_days=1,
    )

    assert assessment is not None
    assert assessment.exit_reason == expected_reason
    assert assessment.strategy_return == pytest.approx(expected_return)
    assert assessment.status is expected_status


def test_open_trade_time_stop_uses_configured_future_session_close() -> None:
    assessment = assess_trade(
        _record(
            trade_intent="open_long",
            entry_price=100.0,
            entry_price_low=99.0,
            entry_price_high=101.0,
            target_price=120.0,
            stop_loss=80.0,
            time_stop_days=3,
        ),
        _execution_bars(
            [
                (100.0, 101.0, 99.0, 100.0),
                (101.0, 103.0, 100.0, 102.0),
                (102.0, 106.0, 101.0, 105.0),
                (105.0, 111.0, 104.0, 110.0),
                (110.0, 116.0, 109.0, 115.0),
            ]
        ),
        horizon_days=5,
    )

    assert assessment is not None
    assert assessment.exit_reason == "time_stop"
    assert assessment.strategy_return == pytest.approx(0.05)
    assert assessment.status is AssessmentStatus.CORRECT


@pytest.mark.parametrize("trade_intent", ["reduce_long", "close_long"])
@pytest.mark.parametrize(
    ("horizon_close", "expected_status"),
    [(90.0, AssessmentStatus.CORRECT), (110.0, AssessmentStatus.INCORRECT)],
)
def test_long_exit_measures_avoided_loss_or_opportunity_cost(
    trade_intent: str,
    horizon_close: float,
    expected_status: AssessmentStatus,
) -> None:
    assessment = assess_trade(
        _record(
            trade_intent=trade_intent,
            entry_price=101.0,
            entry_price_low=100.0,
            entry_price_high=102.0,
        ),
        _execution_bars([(100.0, 100.5, 99.0, horizon_close)]),
        horizon_days=1,
    )

    assert assessment is not None
    assert assessment.entry_triggered is True
    assert assessment.entry_price == 100.5
    assert assessment.exit_reason == "execution"
    assert assessment.strategy_return == pytest.approx((100.5 - horizon_close) / 100.5)
    assert assessment.status is expected_status


def test_reduce_long_can_fill_after_residual_position_time_stop() -> None:
    assessment = assess_trade(
        _record(
            trade_intent="reduce_long",
            position_advice="light",
            entry_price=105.0,
            entry_price_low=104.0,
            entry_price_high=106.0,
            time_stop_days=3,
        ),
        _execution_bars(
            [
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (104.0, 106.0, 103.0, 105.0),
                (90.0, 91.0, 89.0, 90.0),
            ]
        ),
        horizon_days=5,
    )

    assert assessment is not None
    assert assessment.entry_triggered is True
    assert assessment.entry_price == 105.0
    assert any("not simulated" in limitation for limitation in assessment.limitations)


@pytest.mark.parametrize(
    ("horizon_close", "expected_status"),
    [(110.0, AssessmentStatus.CORRECT), (90.0, AssessmentStatus.INCORRECT)],
)
def test_cover_short_measures_avoided_loss_or_opportunity_cost(
    horizon_close: float,
    expected_status: AssessmentStatus,
) -> None:
    assessment = assess_trade(
        _record(
            direction="bearish",
            trade_intent="cover_short",
            entry_price=100.0,
            entry_price_low=99.0,
            entry_price_high=101.0,
        ),
        _execution_bars([(100.0, 101.0, 99.0, horizon_close)]),
        horizon_days=1,
    )

    assert assessment is not None
    assert assessment.entry_triggered is True
    assert assessment.entry_price == 100.0
    assert assessment.exit_reason == "execution"
    assert assessment.strategy_return == pytest.approx((horizon_close - 100.0) / 100.0)
    assert assessment.status is expected_status


def test_historical_missing_or_inferred_intent_never_fakes_strategy_return() -> None:
    for record in (
        _record(),
        _record(trade_intent="open_long", intent_inferred=True),
    ):
        assessment = assess_trade(record, _constant_atr_bars(103.0), horizon_days=1)

        assert assessment is not None
        assert assessment.strategy_return is None
        assert assessment.entry_triggered is False
        assert assessment.limitations


def test_hold_is_direction_only_and_does_not_fake_zero_return() -> None:
    assessment = assess_trade(
        _record(trade_intent="hold", position_advice="none"),
        _constant_atr_bars(103.0),
        horizon_days=1,
    )

    assert assessment is not None
    assert assessment.status is AssessmentStatus.CORRECT
    assert assessment.entry_triggered is False
    assert assessment.strategy_return is None
    assert "hold" in " ".join(assessment.limitations).lower()
