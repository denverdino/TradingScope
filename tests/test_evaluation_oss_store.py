"""Tests for strict schema-v2 evaluation ingestion."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from tests.test_output_models import _all_outputs
from tradingscope.agents.evaluation.oss_store import (
    OSSAnalysisStore,
    build_record_from_portfolio,
)
from tradingscope.agents.output import PortfolioManagerOutput


def test_build_record_requires_valid_v2_portfolio() -> None:
    data = _all_outputs()[-1].model_dump(mode="json")
    data["trade_intent"] = "open_long"
    data["position_advice"] = "medium"
    data["time_stop_days"] = 3
    data["price_plan"].update(
        {
            "entry_price": 101.0,
            "entry_price_low": 100.0,
            "entry_price_high": 102.0,
        },
    )
    portfolio = PortfolioManagerOutput.model_validate(data)
    record = build_record_from_portfolio(portfolio)

    assert record.action == portfolio.decision.action.value
    assert record.confidence == portfolio.decision.confidence
    assert record.reasoning == "；".join(portfolio.decision.reasoning)[:100]
    assert record.trade_intent == "open_long"
    assert record.entry_price_low == 100.0
    assert record.entry_price_high == 102.0
    assert record.position_advice == "medium"
    assert record.time_stop_days == 3
    assert record.intent_inferred is False


def test_build_record_marks_missing_portfolio_intent_as_inferred() -> None:
    record = build_record_from_portfolio(_all_outputs()[-1])

    assert record.trade_intent is None
    assert record.intent_inferred is True


def test_malformed_v2_portfolio_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PortfolioManagerOutput.model_validate({"schema_version": "2.0"})


@pytest.mark.asyncio
async def test_store_reads_only_manifest_completed_v2_output(tmp_path) -> None:
    portfolio = _all_outputs()[-1]
    fetch = AsyncMock(return_value=portfolio.model_dump(mode="json"))

    with patch(
        "tradingscope.agents.evaluation.oss_store.async_fetch_completed_v2_output",
        fetch,
    ):
        records = await OSSAnalysisStore(str(tmp_path)).list_pending(
            tickers=["AAPL"],
            date="2026-07-14",
        )

    fetch.assert_awaited_once_with("2026-07-14", "AAPL")
    assert records[0].ticker == "AAPL"


def test_horizon_state_distinguishes_each_horizon_and_preserves_legacy_day_one(tmp_path) -> None:
    store = OSSAnalysisStore(str(tmp_path))

    assert store.mark_evaluated("AAPL", "2026-07-20", 3)
    assert store.is_evaluated("AAPL", "2026-07-20", 3)
    assert not store.is_evaluated("AAPL", "2026-07-20", 5)

    store._save_evaluated_set({"2026-07-20/AAPL"})

    assert store.is_evaluated("AAPL", "2026-07-20", 1)
    assert not store.is_evaluated("AAPL", "2026-07-20", 3)


@pytest.mark.parametrize("horizon_days", [0, 2, 4, -1, True, False, 1.0, 3.0, 5.0])
def test_store_rejects_invalid_horizon_when_checking_evaluation_state(tmp_path, horizon_days) -> None:
    store = OSSAnalysisStore(str(tmp_path))

    with pytest.raises(ValueError, match=r"allowed horizons.*1, 3, 5"):
        store.is_evaluated("AAPL", "2026-07-20", horizon_days)


@pytest.mark.parametrize("horizon_days", [0, 2, 4, -1, True, False, 1.0, 3.0, 5.0])
def test_store_rejects_invalid_horizon_without_writing_tracking_file(tmp_path, horizon_days) -> None:
    store = OSSAnalysisStore(str(tmp_path))

    with pytest.raises(ValueError, match=r"allowed horizons.*1, 3, 5"):
        store.mark_evaluated("AAPL", "2026-07-20", horizon_days)

    assert not (tmp_path / "oss_evaluated.json").exists()


@pytest.mark.asyncio
async def test_store_skips_report_only_after_all_evaluation_horizons_complete(tmp_path) -> None:
    store = OSSAnalysisStore(str(tmp_path))
    for horizon_days in (1, 3, 5):
        assert store.mark_evaluated("AAPL", "2026-07-14", horizon_days)

    with patch(
        "tradingscope.agents.evaluation.oss_store.async_fetch_completed_v2_output",
        AsyncMock(),
    ) as fetch:
        records = await store.list_pending(tickers=["AAPL"], date="2026-07-14")

    assert records == []
    fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_keeps_report_pending_when_only_one_horizon_is_complete(tmp_path) -> None:
    portfolio = _all_outputs()[-1]
    store = OSSAnalysisStore(str(tmp_path))
    assert store.mark_evaluated("AAPL", "2026-07-14", 1)

    with patch(
        "tradingscope.agents.evaluation.oss_store.async_fetch_completed_v2_output",
        AsyncMock(return_value=portfolio.model_dump(mode="json")),
    ) as fetch:
        records = await store.list_pending(tickers=["AAPL"], date="2026-07-14")

    fetch.assert_awaited_once_with("2026-07-14", "AAPL")
    assert [record.ticker for record in records] == ["AAPL"]


def test_default_candidates_cover_the_prior_fourteen_calendar_dates() -> None:
    candidates = OSSAnalysisStore._generate_date_candidates("2026-07-20", "AAPL")

    assert candidates == [
        ("2026-07-19", "AAPL"),
        ("2026-07-18", "AAPL"),
        ("2026-07-17", "AAPL"),
        ("2026-07-16", "AAPL"),
        ("2026-07-15", "AAPL"),
        ("2026-07-14", "AAPL"),
        ("2026-07-13", "AAPL"),
        ("2026-07-12", "AAPL"),
        ("2026-07-11", "AAPL"),
        ("2026-07-10", "AAPL"),
        ("2026-07-09", "AAPL"),
        ("2026-07-08", "AAPL"),
        ("2026-07-07", "AAPL"),
        ("2026-07-06", "AAPL"),
    ]
