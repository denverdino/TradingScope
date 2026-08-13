"""Tests for atomic schema-v2 OSS persistence."""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.test_output_models import _all_outputs
from tradingscope.agents.output import AnalysisResult, AnalystOutputs
from tradingscope.utils import oss_report_uploader
from tradingscope.utils import oss_structured_output_uploader as structured_uploader
from tradingscope.utils.oss_structured_output_uploader import (
    analysis_persistence_session,
    persist_analysis_result,
    persist_node_output,
    prepare_analysis_persistence,
)


@pytest.fixture(autouse=True)
def local_results_dir(tmp_path):
    with patch(
        "tradingscope.utils.oss_structured_output_uploader.DEFAULT_CONFIG",
        {"results_dir": str(tmp_path)},
    ):
        yield


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
async def test_prepare_analysis_persistence_removes_stale_local_and_oss_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "data" / "2026-07-14" / "AAPL" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"status":"complete"}')
    client = MagicMock()

    with (
        patch("tradingscope.utils.oss_structured_output_uploader._get_client", return_value=client),
        patch("tradingscope.utils.oss_structured_output_uploader._get_bucket_name", return_value="bucket"),
    ):
        await prepare_analysis_persistence("2026-07-14", "AAPL")

    assert not manifest_path.exists()
    request = client.delete_object.call_args.args[0]
    assert request.bucket == "bucket"
    assert request.key == "tradingscope/2026-07-14/AAPL/manifest.json"


@pytest.mark.asyncio
async def test_persistence_session_serializes_runs_for_the_same_ticker_and_date() -> None:
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    entries: list[str] = []

    async def run(label: str) -> None:
        async with analysis_persistence_session("2026-07-14", "AAPL"):
            entries.append(label)
            if label == "first":
                first_entered.set()
                await release_first.wait()

    with patch(
        "tradingscope.utils.oss_structured_output_uploader.prepare_analysis_persistence",
        new=AsyncMock(),
    ):
        first = asyncio.create_task(run("first"))
        await first_entered.wait()
        second = asyncio.create_task(run("second"))
        await asyncio.sleep(0)
        assert entries == ["first"]
        release_first.set()
        await asyncio.gather(first, second)

    assert entries == ["first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "async_name", "blocking_name"),
    [
        (structured_uploader, "upload_structured_output", "_upload_structured_output"),
        (oss_report_uploader, "upload_report", "_upload_report"),
    ],
)
async def test_cancelled_oss_upload_drains_blocking_operation(module, async_name, blocking_name) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_upload(*_args):
        started.set()
        release.wait(timeout=1)
        return True

    with patch.object(module, blocking_name, side_effect=blocking_upload):
        upload = getattr(module, async_name)
        task = asyncio.create_task(upload("2026-07-14", "AAPL", "market_analyst", "content"))
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        try:
            assert not task.done()
        finally:
            release.set()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_cancelled_manifest_invalidation_holds_same_key_session_lock(tmp_path) -> None:
    delete_started = threading.Event()
    release_delete = threading.Event()
    second_entered = asyncio.Event()
    client = MagicMock()

    def delete_object(_request):
        if not delete_started.is_set():
            delete_started.set()
            release_delete.wait(timeout=1)

    client.delete_object.side_effect = delete_object

    async def first_run() -> None:
        async with analysis_persistence_session("2026-07-14", "AAPL"):
            pass

    async def second_run() -> None:
        async with analysis_persistence_session("2026-07-14", "AAPL"):
            second_entered.set()

    with (
        patch("tradingscope.utils.oss_structured_output_uploader._get_client", return_value=client),
        patch("tradingscope.utils.oss_structured_output_uploader._get_bucket_name", return_value="bucket"),
    ):
        first = asyncio.create_task(first_run())
        while not delete_started.is_set():
            await asyncio.sleep(0)
        first.cancel()
        second = asyncio.create_task(second_run())
        await asyncio.sleep(0)
        try:
            assert not second_entered.is_set()
        finally:
            release_delete.set()
            with suppress(asyncio.CancelledError):
                await first
            await second


@pytest.mark.asyncio
async def test_local_persistence_rejects_unsafe_ticker_path(analysis_result: AnalysisResult) -> None:
    with patch("tradingscope.utils.oss_structured_output_uploader._get_client", return_value=None):
        with pytest.raises(ValueError, match="path component"):
            await persist_node_output(
                "market_analyst",
                analysis_result.analysts.market,
                ticker="../AAPL",
                trade_date="2026-07-14",
            )


@pytest.mark.asyncio
async def test_persistence_writes_manifest_last(analysis_result: AnalysisResult, tmp_path) -> None:
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
    manifest_path = tmp_path / "data" / "2026-07-14" / "AAPL" / "manifest.json"
    assert json.loads(manifest_path.read_text())["status"] == "complete"


@pytest.mark.asyncio
async def test_node_persistence_writes_local_json_and_markdown_without_oss(
    analysis_result: AnalysisResult,
    tmp_path,
) -> None:
    output = analysis_result.analysts.market

    with (
        patch(
            "tradingscope.utils.oss_structured_output_uploader.DEFAULT_CONFIG",
            {"results_dir": str(tmp_path)},
        ),
        patch("tradingscope.utils.oss_structured_output_uploader._get_client", return_value=None),
    ):
        await persist_node_output(
            "market_analyst",
            output,
            ticker="AAPL",
            trade_date="2026-07-14",
        )

    output_dir = tmp_path / "data" / "2026-07-14" / "AAPL"
    assert json.loads((output_dir / "market_analyst.json").read_text()) == output.model_dump(mode="json")
    assert "技术面分析" in (output_dir / "market_analyst.md").read_text()


@pytest.mark.asyncio
async def test_node_persistence_uploads_json_and_markdown_when_oss_is_configured(
    analysis_result: AnalysisResult,
    tmp_path,
) -> None:
    output = analysis_result.trader
    upload_json = AsyncMock(return_value={"trader": True})
    upload_markdown = AsyncMock(return_value={"trader": True})

    with (
        patch(
            "tradingscope.utils.oss_structured_output_uploader.DEFAULT_CONFIG",
            {"results_dir": str(tmp_path)},
        ),
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
        await persist_node_output(
            "trader",
            output,
            ticker="AAPL",
            trade_date="2026-07-14",
        )

    assert list(upload_json.await_args.args[2]) == ["trader"]
    assert list(upload_markdown.await_args.args[2]) == ["trader"]


@pytest.mark.asyncio
async def test_node_persistence_fails_when_configured_oss_upload_is_incomplete(
    analysis_result: AnalysisResult,
    tmp_path,
) -> None:
    output = analysis_result.trader

    with (
        patch("tradingscope.utils.oss_structured_output_uploader._get_client", return_value=object()),
        patch(
            "tradingscope.utils.oss_structured_output_uploader.upload_structured_outputs",
            new=AsyncMock(return_value={"trader": True}),
        ),
        patch(
            "tradingscope.utils.oss_structured_output_uploader.upload_reports",
            new=AsyncMock(return_value={"trader": False}),
        ),
    ):
        with pytest.raises(RuntimeError, match="trader.md"):
            await persist_node_output(
                "trader",
                output,
                ticker="AAPL",
                trade_date="2026-07-14",
            )

    output_dir = tmp_path / "data" / "2026-07-14" / "AAPL"
    assert (output_dir / "trader.json").is_file()
    assert (output_dir / "trader.md").is_file()


@pytest.mark.asyncio
async def test_persistence_serializes_full_v2_result(analysis_result: AnalysisResult) -> None:
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

    assert json.loads(captured["full_report"])["schema_version"] == "2.0"


@pytest.mark.asyncio
async def test_persistence_omits_manifest_after_failure(analysis_result: AnalysisResult, tmp_path) -> None:
    upload_json = AsyncMock(return_value={"full_report": False})
    upload_markdown = AsyncMock(return_value={"full_report": True})

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
        with pytest.raises(RuntimeError, match="full_report.json"):
            await persist_analysis_result(analysis_result)

    assert upload_json.await_count == 1
    assert not (tmp_path / "data" / "2026-07-14" / "AAPL" / "manifest.json").exists()
