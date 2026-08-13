"""Upload agent structured outputs (JSON) to Alibaba Cloud OSS.

Structured outputs are stored at: tradingscope/<date>/<ticker>/<agent_name>.json

Environment variables:
    OSS_ACCESS_KEY_ID     - Alibaba Cloud access key ID
    OSS_ACCESS_KEY_SECRET - Alibaba Cloud access key secret
    OSS_REGION            - OSS region (e.g. cn-hangzhou)
    OSS_BUCKET            - OSS bucket name
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from tradingscope.agents.output import AgentOutputBase, AnalysisResult
from tradingscope.agents.renderers import render_full_report, render_markdown
from tradingscope.default_config import DEFAULT_CONFIG
from tradingscope.utils.oss_report_uploader import _run_blocking_oss_operation, upload_reports

logger = logging.getLogger(__name__)

_OSS_KEY_PREFIX = "tradingscope"


@dataclass
class _PersistenceLock:
    lock: asyncio.Lock
    users: int = 0


_persistence_locks: dict[tuple[str, str], _PersistenceLock] = {}


def _get_client():
    """Lazily initialize and return the OSS client (shared with report uploader)."""
    from tradingscope.utils.oss_report_uploader import _get_client as _get_md_client

    return _get_md_client()


def _get_bucket_name():
    """Return the configured OSS bucket name (shared with report uploader)."""
    from tradingscope.utils.oss_report_uploader import _bucket_name as _md_bucket

    return _md_bucket


def _upload_structured_output(trade_date: str, ticker: str, agent_name: str, content: str) -> bool:
    """Upload a single structured output (JSON) to OSS (blocking).

    Args:
        trade_date: Trading date, e.g. "2026-02-16"
        ticker: Stock ticker, e.g. "MSFT"
        agent_name: Agent name, e.g. "market_analyst"
        content: JSON content as string

    Returns:
        True if upload succeeded, False otherwise.
    """
    import alibabacloud_oss_v2 as oss

    client = _get_client()
    bucket_name = _get_bucket_name()
    if client is None or bucket_name is None:
        return False

    key = f"{_OSS_KEY_PREFIX}/{trade_date}/{ticker}/{agent_name}.json"
    try:
        client.put_object(
            oss.PutObjectRequest(
                bucket=bucket_name,
                key=key,
                body=content.encode("utf-8"),
                content_type="application/json; charset=utf-8",
            )
        )
        logger.info(f"[OSSStructuredUploader] Uploaded: oss://{bucket_name}/{key}")
        return True
    except Exception as e:
        logger.warning(f"[OSSStructuredUploader] Failed to upload {key}: {e}")
        return False


async def upload_structured_output(trade_date: str, ticker: str, agent_name: str, content: str) -> bool:
    """Upload a single structured output to OSS (async wrapper)."""
    return await _run_blocking_oss_operation(_upload_structured_output, trade_date, ticker, agent_name, content)


async def upload_structured_outputs(trade_date: str, ticker: str, outputs: Dict[str, str]) -> Dict[str, bool]:
    """Upload multiple agent structured outputs to OSS concurrently.

    Args:
        trade_date: Trading date, e.g. "2026-02-16"
        ticker: Stock ticker, e.g. "MSFT"
        outputs: Mapping of agent_name -> JSON content string

    Returns:
        Mapping of agent_name -> upload success/failure.
    """
    if not outputs:
        return {}

    if _get_client() is None:
        return dict.fromkeys(outputs, False)

    tasks = {name: upload_structured_output(trade_date, ticker, name, content) for name, content in outputs.items() if content}
    results = {}
    for name, task in tasks.items():
        results[name] = await task
    return results


def _local_output_dir(trade_date: str, ticker: str) -> Path:
    if not ticker or ticker in {".", ".."} or "/" in ticker or "\\" in ticker:
        raise ValueError("ticker must be a single safe path component")
    return Path(DEFAULT_CONFIG["results_dir"]) / "data" / trade_date / ticker


def _write_local_artifacts(trade_date: str, ticker: str, outputs: Dict[str, str]) -> None:
    output_dir = _local_output_dir(trade_date, ticker)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in outputs.items():
        (output_dir / filename).write_text(content, encoding="utf-8")


def _delete_oss_manifest(client, bucket_name: str, trade_date: str, ticker: str) -> None:
    import alibabacloud_oss_v2 as oss

    key = f"{_OSS_KEY_PREFIX}/{trade_date}/{ticker}/manifest.json"
    try:
        client.delete_object(
            oss.DeleteObjectRequest(
                bucket=bucket_name,
                key=key,
            ),
        )
        logger.info("[OSSStructuredUploader] Invalidated: oss://%s/%s", bucket_name, key)
    except Exception as exc:
        raise RuntimeError(f"failed to invalidate completion manifest: {key}") from exc


async def prepare_analysis_persistence(trade_date: str, ticker: str) -> None:
    """Invalidate a prior completion marker before writing a new run."""
    (_local_output_dir(trade_date, ticker) / "manifest.json").unlink(missing_ok=True)

    client = _get_client()
    if client is None:
        return
    bucket_name = _get_bucket_name()
    if bucket_name is None:
        raise RuntimeError("failed to invalidate completion manifest: OSS bucket is unavailable")
    await _run_blocking_oss_operation(
        _delete_oss_manifest,
        client,
        bucket_name,
        trade_date,
        ticker,
    )


async def _upload_json_and_markdown(
    trade_date: str,
    ticker: str,
    json_outputs: Dict[str, str],
    markdown_outputs: Dict[str, str],
) -> tuple[Dict[str, bool], Dict[str, bool]]:
    tasks = [
        asyncio.create_task(upload_structured_outputs(trade_date, ticker, json_outputs)),
        asyncio.create_task(upload_reports(trade_date, ticker, markdown_outputs)),
    ]
    try:
        json_results, markdown_results = await asyncio.gather(*tasks)
        return json_results, markdown_results
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


@asynccontextmanager
async def analysis_persistence_session(trade_date: str, ticker: str) -> AsyncIterator[None]:
    """Serialize and initialize runs that target the same artifact directory."""
    key = (trade_date, ticker)
    state = _persistence_locks.get(key)
    if state is None:
        state = _PersistenceLock(lock=asyncio.Lock())
        _persistence_locks[key] = state
    state.users += 1
    try:
        async with state.lock:
            await prepare_analysis_persistence(trade_date, ticker)
            yield
    finally:
        state.users -= 1
        if state.users == 0:
            _persistence_locks.pop(key, None)


async def persist_node_output(
    agent_name: str,
    output: AgentOutputBase,
    *,
    ticker: str,
    trade_date: str,
) -> None:
    """Persist one validated workflow node locally and, when configured, to OSS."""
    json_content = output.model_dump_json(indent=2)
    markdown_content = render_markdown(output)
    _write_local_artifacts(
        trade_date,
        ticker,
        {
            f"{agent_name}.json": json_content,
            f"{agent_name}.md": markdown_content,
        },
    )

    if _get_client() is None:
        logger.info("[OSSStructuredUploader] OSS not configured; persisted %s locally", agent_name)
        return

    json_results, markdown_results = await _upload_json_and_markdown(
        trade_date,
        ticker,
        {agent_name: json_content},
        {agent_name: markdown_content},
    )
    failed = []
    if not json_results.get(agent_name, False):
        failed.append(f"{agent_name}.json")
    if not markdown_results.get(agent_name, False):
        failed.append(f"{agent_name}.md")
    if failed:
        raise RuntimeError(f"failed to persist required artifacts: {', '.join(failed)}")


async def persist_analysis_result(result: AnalysisResult) -> None:
    """Persist the completed report and write its manifest last.

    Individual workflow nodes are persisted as they finish. OSS remains
    optional; when configured, the full report must upload successfully before
    the completion manifest is written.
    """
    trade_date = result.trade_date.isoformat()
    full_report_json = result.model_dump_json(indent=2)
    full_report_markdown = render_full_report(result)
    _write_local_artifacts(
        trade_date,
        result.ticker,
        {
            "full_report.json": full_report_json,
            "full_report.md": full_report_markdown,
        },
    )

    manifest = json.dumps(
        {
            "schema_version": "2.0",
            "status": "complete",
            "ticker": result.ticker,
            "trade_date": trade_date,
        },
        ensure_ascii=False,
        indent=2,
    )

    if _get_client() is None:
        _write_local_artifacts(trade_date, result.ticker, {"manifest.json": manifest})
        logger.info("[OSSStructuredUploader] OSS not configured; persisted completed result locally")
        return

    json_results, markdown_results = await _upload_json_and_markdown(
        trade_date,
        result.ticker,
        {"full_report": full_report_json},
        {"full_report": full_report_markdown},
    )
    failed = []
    if not json_results.get("full_report", False):
        failed.append("full_report.json")
    if not markdown_results.get("full_report", False):
        failed.append("full_report.md")
    if failed:
        raise RuntimeError(f"failed to persist required artifacts: {', '.join(failed)}")

    manifest_result = await upload_structured_outputs(
        trade_date,
        result.ticker,
        {"manifest": manifest},
    )
    if not manifest_result.get("manifest", False):
        raise RuntimeError("failed to persist completion manifest")
    _write_local_artifacts(trade_date, result.ticker, {"manifest.json": manifest})
