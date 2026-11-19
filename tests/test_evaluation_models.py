"""Tests for strict persisted evaluation records."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tests.support.evaluation import EVALUATED_AT, completed_run, evaluation_record, flat_future, market_bars
from tradingscope.agents.evaluation.evaluator import build_evaluation_record
from tradingscope.agents.evaluation.models import AnalysisRecord, EvaluationRecord, effective_return_basis
from tradingscope.agents.output import TradeIntent


@pytest.mark.parametrize("intent", list(TradeIntent))
@pytest.mark.parametrize("explicit_basis", [None, "", "explicit"])
def test_effective_return_basis_preserves_precedence_and_fallback(intent, explicit_basis):
    trade = (
        evaluation_record()
        .stage("portfolio_manager")
        .trade.model_copy(
            update={"trade_intent": intent, "strategy_return_basis": explicit_basis},
        )
    )
    expected = explicit_basis or ("position_return" if intent in {"open_long", "open_short"} else "exit_relative_performance")
    assert effective_return_basis(trade) == expected


def test_trade_record_defaults_execution_fields():
    record = AnalysisRecord(analysis_date="2026-07-01", direction="bullish", trade_intent="hold")
    assert record.trade_intent == "hold"
    assert record.time_stop_days is None
    assert record.execution_price is None


def test_evaluation_roundtrip_preserves_results():
    record = evaluation_record()
    assert EvaluationRecord.model_validate_json(record.model_dump_json()) == record
    assert record.evaluation_id("portfolio_manager") == "2026-07-14/AAPL/run-1/h1/portfolio_manager"


def test_evaluation_rejects_unknown_fields():
    data = evaluation_record().model_dump(mode="json")
    data["lesson"] = "not part of deterministic evaluation"
    with pytest.raises(ValidationError):
        EvaluationRecord.model_validate(data)


def test_evaluation_timestamp_is_explicit_not_created_by_the_model():
    record = evaluation_record()
    assert record.evaluated_at == EVALUATED_AT
    data = record.model_dump(mode="json")
    del data["evaluated_at"]
    with pytest.raises(ValidationError):
        EvaluationRecord.model_validate(data)
    timestamp = datetime(2026, 7, 20, tzinfo=timezone.utc)
    data["evaluated_at"] = timestamp
    assert EvaluationRecord.model_validate(data).evaluated_at == timestamp


def test_source_outputs_are_detached_from_mutable_analysis_objects():
    run = completed_run()
    record = build_evaluation_record(
        run,
        market_bars(flat_future([101.0] * 5)),
        "XLK",
        market_bars(flat_future([100.0] * 5)),
        [],
        1,
        evaluated_at=EVALUATED_AT,
    )
    previous = record.source_outputs["portfolio_manager"]["decision"]["summary"]
    run.outputs["portfolio_manager"].decision.summary = "replaced run"
    assert record.source_outputs["portfolio_manager"]["decision"]["summary"] == previous


@pytest.mark.parametrize("value", ["..", "../other", "a/b", "a\\\\b"])
def test_record_rejects_unsafe_run_ids(value):
    data = evaluation_record().model_dump(mode="json")
    data["run_id"] = value
    with pytest.raises(ValidationError):
        EvaluationRecord.model_validate(data)
