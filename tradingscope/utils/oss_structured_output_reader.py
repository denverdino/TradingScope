"""Read agent structured outputs (JSON) from Alibaba Cloud OSS.

Discovers and fetches structured outputs stored at: tradingscope/<date>/<ticker>/<agent_name>.json

Requires OSS SDK configuration:
    OSS_REGION, OSS_BUCKET, OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_OSS_KEY_PREFIX = "tradingscope"


def _get_client_and_bucket():
    """Get OSS client and bucket name from the uploader module."""
    import tradingscope.utils.oss_structured_output_uploader as uploader

    client = uploader._get_client()
    bucket_name = uploader._get_bucket_name()
    return client, bucket_name


def fetch_structured_output(date: str, ticker: str, agent: str = "portfolio_manager") -> Optional[Dict[str, Any]]:
    """Fetch a single structured output's content from OSS via SDK.

    Args:
        date: Trade date (YYYY-MM-DD)
        ticker: Stock ticker (e.g. "AAPL")
        agent: Agent name (default: "portfolio_manager")

    Returns:
        Parsed JSON dict, or None on failure.
    """
    client, bucket_name = _get_client_and_bucket()
    if client is None:
        return None

    try:
        import alibabacloud_oss_v2 as oss
    except ImportError:
        return None

    key = f"{_OSS_KEY_PREFIX}/{date}/{ticker}/{agent}.json"
    try:
        resp = client.get_object(
            oss.GetObjectRequest(
                bucket=bucket_name,
                key=key,
            )
        )
        content = resp.body.content
        if isinstance(content, bytes):
            raw = content.decode("utf-8")
        else:
            raw = content
        return json.loads(raw)
    except Exception as e:
        logger.debug("[OSSStructuredReader] Failed to fetch %s: %s", key, e)
        return None


async def async_fetch_structured_output(date: str, ticker: str, agent: str = "portfolio_manager") -> Optional[Dict[str, Any]]:
    """Async wrapper for fetch_structured_output."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fetch_structured_output, date, ticker, agent)