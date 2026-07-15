"""Tests for atomic schema-v2 OSS persistence."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from tests.test_output_models import _all_outputs
from tradingscope.agents.output import AnalysisResult, AnalystOutputs
from tradingscope.utils.oss_structured_output_uploader import persist_analysis_result


@pytest.fixture
def analysis_result() -> AnalysisResult:
    market, fundamentals, news, social, research, trader, portfolio = _all_outputs()
    return AnalysisResult(
        schema_version="2.0",
        ticker="AAPL",
        trade_date="2026-07-14",
        latest_trading_date="2026-07-13",
        analysts=AnalystOutputs(
            market=market,
            fundamentals=fundamentals,
            news=news,
            social_media=social,
        ),
        research_manager=research,
        trader=trader,
        portfolio_manager=portfolio,
    )


@pytest.mark.asyncio
async def test_persistence_writes_manifest_last(analysis_result: AnalysisResult) -> None:
    upload_json = AsyncMock(side_effect=lambda _date, _ticker, values: dict.fromkeys(values, True))
    upload_markdown = AsyncMock(side_effect=lambda _date, _ticker, values: dict.fromkeys(values, True))

    with (
        patch("tradingscope.utils.oss_structured_output_uploader._get_client", return_value=object()),
        patch(
            "tradingscope.utils.oss_structured_output_uploader.upload_structured_outputs",
            upload_json,
        ),
        patch(
            "tradingscope.utils.oss_structured_output_uploader.upload_reports",
            upload_markdown,
        ),
    ):
        await persist_analysis_result(analysis_result)

    assert list(upload_json.await_args_list[-1].args[2]) == ["manifest"]
    assert upload_json.await_count == 2


@pytest.mark.asyncio
async def test_persistence_serializes_each_v2_model(analysis_result: AnalysisResult) -> None:
    captured: dict[str, str] = {}

    async def capture(_date: str, _ticker: str, values: dict[str, str]):
        captured.update(values)
        return dict.fromkeys(values, True)

    with (
        patch("tradingscope.utils.oss_structured_output_uploader._get_client", return_value=object()),
        patch(
            "tradingscope.utils.oss_structured_output_uploader.upload_structured_outputs",
            side_effect=capture,
        ),
        patch(
            "tradingscope.utils.oss_structured_output_uploader.upload_reports",
            side_effect=lambda _date, _ticker, values: dict.fromkeys(values, True),
        ),
    ):
        await persist_analysis_result(analysis_result)

    for name in (
        "market_analyst",
        "fundamentals_analyst",
        "news_analyst",
        "social_media_analyst",
        "research_manager",
        "trader",
        "portfolio_manager",
        "full_report",
    ):
        assert json.loads(captured[name])["schema_version"] == "2.0"


@pytest.mark.asyncio
async def test_persistence_omits_manifest_after_failure(analysis_result: AnalysisResult) -> None:
    upload_json = AsyncMock(return_value={"market_analyst": False})
    upload_markdown = AsyncMock(return_value={})

    with (
        patch("tradingscope.utils.oss_structured_output_uploader._get_client", return_value=object()),
        patch(
            "tradingscope.utils.oss_structured_output_uploader.upload_structured_outputs",
            upload_json,
        ),
        patch(
            "tradingscope.utils.oss_structured_output_uploader.upload_reports",
            upload_markdown,
        ),
    ):
        with pytest.raises(RuntimeError, match="market_analyst"):
            await persist_analysis_result(analysis_result)

    assert upload_json.await_count == 1
