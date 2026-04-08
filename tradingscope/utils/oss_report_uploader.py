"""Upload agent reports to Alibaba Cloud OSS.

Reports are stored at: tradingscope/<date>/<ticker>/<agent_name>.md

Environment variables:
    OSS_ACCESS_KEY_ID     - Alibaba Cloud access key ID
    OSS_ACCESS_KEY_SECRET - Alibaba Cloud access key secret
    OSS_REGION            - OSS region (e.g. cn-hangzhou)
    OSS_BUCKET            - OSS bucket name
"""

import asyncio
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

_client = None
_bucket_name = None


def _get_client():
    """Lazily initialize and return the OSS client."""
    global _client, _bucket_name
    if _client is not None:
        return _client

    try:
        import alibabacloud_oss_v2 as oss
    except ImportError:
        logger.warning("[OSSUploader] alibabacloud-oss-v2 not installed. Run: pip install alibabacloud-oss-v2")
        return None

    region = os.getenv("OSS_REGION")
    _bucket_name = os.getenv("OSS_BUCKET")

    if not region or not _bucket_name:
        logger.warning("[OSSUploader] OSS not configured. Set OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_REGION, OSS_BUCKET.")
        return None

    try:
        cred_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
    except oss.credentials.CredentialsEmptyError:
        logger.warning("[OSSUploader] OSS credentials not set. Set OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET.")
        return None

    cfg = oss.Config(
        region=region,
        credentials_provider=cred_provider,
    )
    _client = oss.Client(cfg)
    return _client


def _upload_report(trade_date: str, ticker: str, agent_name: str, content: str) -> bool:
    """Upload a single report to OSS (blocking).

    Args:
        trade_date: Trading date, e.g. "2026-02-16"
        ticker: Stock ticker, e.g. "MSFT"
        agent_name: Agent name, e.g. "market_analyst"
        content: Markdown report content

    Returns:
        True if upload succeeded, False otherwise.
    """
    import alibabacloud_oss_v2 as oss

    client = _get_client()
    if client is None:
        return False

    key = f"tradingscope/{trade_date}/{ticker}/{agent_name}.md"
    try:
        client.put_object(
            oss.PutObjectRequest(
                bucket=_bucket_name,
                key=key,
                body=content.encode("utf-8"),
                content_type="text/markdown; charset=utf-8",
            )
        )
        logger.info(f"[OSSUploader] Uploaded: oss://{_bucket_name}/{key}")
        return True
    except Exception as e:
        logger.warning(f"[OSSUploader] Failed to upload {key}: {e}")
        return False


async def upload_report(trade_date: str, ticker: str, agent_name: str, content: str) -> bool:
    """Upload a single report to OSS (async wrapper)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _upload_report, trade_date, ticker, agent_name, content)


async def upload_reports(trade_date: str, ticker: str, reports: Dict[str, str]) -> Dict[str, bool]:
    """Upload multiple agent reports to OSS concurrently.

    Args:
        trade_date: Trading date, e.g. "2026-02-16"
        ticker: Stock ticker, e.g. "MSFT"
        reports: Mapping of agent_name -> report content

    Returns:
        Mapping of agent_name -> upload success/failure.
    """
    if not reports:
        return {}

    if _get_client() is None:
        return dict.fromkeys(reports, False)

    tasks = {name: upload_report(trade_date, ticker, name, content) for name, content in reports.items() if content}
    results = {}
    for name, task in tasks.items():
        results[name] = await task
    return results
