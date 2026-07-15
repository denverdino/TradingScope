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
    portfolio = _all_outputs()[-1]
    record = build_record_from_portfolio(portfolio)

    assert record.action == portfolio.decision.action.value
    assert record.confidence == portfolio.decision.confidence
    assert record.reasoning == "；".join(portfolio.decision.reasoning)[:100]


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
