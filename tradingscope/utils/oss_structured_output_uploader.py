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
from typing import Dict

from tradingscope.agents.output import AnalysisResult
from tradingscope.agents.renderers import render_full_report, render_markdown
from tradingscope.utils.oss_report_uploader import upload_reports

logger = logging.getLogger(__name__)

_OSS_KEY_PREFIX = "tradingscope"


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
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _upload_structured_output, trade_date, ticker, agent_name, content)


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


async def persist_analysis_result(result: AnalysisResult) -> None:
    """Persist a complete schema-v2 result and write its manifest last.

    OSS remains optional. When it is configured, every JSON and Markdown
    artifact must succeed before the completion manifest is written.
    """
    if _get_client() is None:
        logger.info("[OSSStructuredUploader] OSS not configured; skipping persistence")
        return

    trade_date = result.trade_date.isoformat()
    node_outputs = {
        "market_analyst": result.analysts.market,
        "fundamentals_analyst": result.analysts.fundamentals,
        "news_analyst": result.analysts.news,
        "social_media_analyst": result.analysts.social_media,
        "research_manager": result.research_manager,
        "trader": result.trader,
        "portfolio_manager": result.portfolio_manager,
        "full_report": result,
    }
    json_outputs = {name: output.model_dump_json(indent=2) for name, output in node_outputs.items()}
    markdown_outputs = {name: render_markdown(output) for name, output in node_outputs.items() if name != "full_report"}
    markdown_outputs["full_report"] = render_full_report(result)

    json_results = await upload_structured_outputs(
        trade_date,
        result.ticker,
        json_outputs,
    )
    markdown_results = await upload_reports(
        trade_date,
        result.ticker,
        markdown_outputs,
    )
    failed = [f"{name}.json" for name in json_outputs if not json_results.get(name, False)]
    failed.extend(f"{name}.md" for name in markdown_outputs if not markdown_results.get(name, False))
    if failed:
        raise RuntimeError(f"failed to persist required artifacts: {', '.join(failed)}")

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
    manifest_result = await upload_structured_outputs(
        trade_date,
        result.ticker,
        {"manifest": manifest},
    )
    if not manifest_result.get("manifest", False):
        raise RuntimeError("failed to persist completion manifest")
