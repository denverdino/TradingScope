from tradingscope.agents.evaluation.models import AnalysisRecord, EvaluationResult


def test_historical_analysis_record_defaults_execution_fields() -> None:
    historical = {
        "ticker": "AAPL",
        "trade_date": "2026-07-01",
        "direction": "bullish",
        "action": "buy",
        "confidence": 0.8,
    }

    record = AnalysisRecord.from_dict(historical)

    assert record.trade_intent is None
    assert record.entry_price_low is None
    assert record.entry_price_high is None
    assert record.position_advice is None
    assert record.time_stop_days is None
    assert record.intent_inferred is False


def test_analysis_record_round_trips_execution_fields() -> None:
    record = AnalysisRecord(
        ticker="TSLA",
        trade_date="2026-07-02",
        direction="bearish",
        action="sell",
        confidence=0.7,
        entry_price=320.0,
        entry_price_low=318.0,
        entry_price_high=322.0,
        target_price=290.0,
        stop_loss=330.0,
        trade_intent="open_short",
        position_advice="light",
        time_stop_days=3,
        intent_inferred=True,
    )

    restored = AnalysisRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.trade_intent == "open_short"
    assert restored.entry_price_low == 318.0
    assert restored.entry_price_high == 322.0
    assert restored.position_advice == "light"
    assert restored.time_stop_days == 3
    assert restored.intent_inferred is True


def test_evaluation_result_keeps_legacy_construction_and_exposes_objective_outcome() -> None:
    legacy = EvaluationResult(
        ticker="AAPL",
        evaluation="方向正确",
        lesson="继续遵守纪律",
    )
    result = EvaluationResult(
        ticker="TSLA",
        evaluation="目标价触发",
        lesson="分批止盈",
        horizon_days=3,
        status="correct",
        entry_triggered=True,
        benchmark_return=0.04,
        strategy_return=0.05,
    )

    assert legacy.horizon_days == 1
    assert legacy.status == "inconclusive"
    assert legacy.entry_triggered is False
    assert legacy.benchmark_return is None
    assert legacy.strategy_return is None
    assert result.horizon_days == 3
    assert result.status == "correct"
    assert result.entry_triggered is True
    assert result.benchmark_return == 0.04
    assert result.strategy_return == 0.05
