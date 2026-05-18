"""Read agent reports from Alibaba Cloud OSS.

Discovers and fetches reports stored at:
  tradingscope/<date>/<ticker>/<agent_name>.md   (markdown)
  tradingscope/<date>/<ticker>/<agent_name>.json (structured JSON)

Requires OSS SDK configuration:
    OSS_REGION, OSS_BUCKET, OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_OSS_KEY_PREFIX = "tradingscope"


def _get_client_and_bucket():
    """Get OSS client and bucket name from the uploader module."""
    import tradingscope.utils.oss_report_uploader as uploader

    client = uploader._get_client()
    return client, uploader._bucket_name


def list_pending_reports(before_date: Optional[str] = None) -> List[Tuple[str, str]]:
    """List all (date, ticker) pairs that have a portfolio_manager.md report.

    Args:
        before_date: Only return reports with date < this value (YYYY-MM-DD).
                     If None, returns all.

    Returns:
        List of (date, ticker) tuples.
    """
    client, bucket_name = _get_client_and_bucket()
    if client is None:
        return []

    try:
        import alibabacloud_oss_v2 as oss
    except ImportError:
        return []

    results: List[Tuple[str, str]] = []

    try:
        # Step 1: List date-level prefixes
        date_prefixes: List[str] = []
        continuation_token = None

        while True:
            resp = client.list_objects_v2(
                oss.ListObjectsV2Request(
                    bucket=bucket_name,
                    prefix=f"{_OSS_KEY_PREFIX}/",
                    delimiter="/",
                    max_keys=1000,
                    continuation_token=continuation_token,
                )
            )

            if resp.common_prefixes:
                for prefix in resp.common_prefixes:
                    # prefix.prefix = "tradingscope/2026-05-04/"
                    date_str = prefix.prefix.rstrip("/").split("/")[-1]
                    if before_date and date_str >= before_date:
                        continue
                    date_prefixes.append(prefix.prefix)

            if not resp.is_truncated:
                break
            continuation_token = resp.next_continuation_token

        # Step 2: For each date, list ticker-level prefixes
        for date_prefix in date_prefixes:
            date_str = date_prefix.rstrip("/").split("/")[-1]
            ticker_token = None

            while True:
                resp = client.list_objects_v2(
                    oss.ListObjectsV2Request(
                        bucket=bucket_name,
                        prefix=date_prefix,
                        delimiter="/",
                        max_keys=1000,
                        continuation_token=ticker_token,
                    )
                )

                if resp.common_prefixes:
                    for prefix in resp.common_prefixes:
                        # prefix.prefix = "tradingscope/2026-05-04/AAPL/"
                        ticker = prefix.prefix.rstrip("/").split("/")[-1]
                        results.append((date_str, ticker))

                if not resp.is_truncated:
                    break
                ticker_token = resp.next_continuation_token

    except Exception as e:
        logger.warning("[OSSReader] Failed to list reports: %s", e)

    return results


def fetch_report(date: str, ticker: str, agent: str = "portfolio_manager") -> Optional[str]:
    """Fetch a single report's content from OSS via SDK.

    Args:
        date: Trade date (YYYY-MM-DD)
        ticker: Stock ticker (e.g. "AAPL")
        agent: Agent name (default: "portfolio_manager")

    Returns:
        Report content as string, or None on failure.
    """
    client, bucket_name = _get_client_and_bucket()
    if client is None:
        return None

    try:
        import alibabacloud_oss_v2 as oss
    except ImportError:
        return None

    key = f"{_OSS_KEY_PREFIX}/{date}/{ticker}/{agent}.md"
    try:
        resp = client.get_object(
            oss.GetObjectRequest(
                bucket=bucket_name,
                key=key,
            )
        )
        content = resp.body.content
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return content
    except Exception as e:
        logger.debug("[OSSReader] Failed to fetch %s: %s", key, e)
        return None


async def async_list_pending_reports(before_date: Optional[str] = None) -> List[Tuple[str, str]]:
    """Async wrapper for list_pending_reports."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, list_pending_reports, before_date)


async def async_fetch_report(date: str, ticker: str, agent: str = "portfolio_manager") -> Optional[str]:
    """Async wrapper for fetch_report."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fetch_report, date, ticker, agent)


def fetch_json_report(date: str, ticker: str, agent: str = "portfolio_manager") -> Optional[Dict[str, Any]]:
    """Fetch a structured JSON report from OSS.

    Reads tradingscope/<date>/<ticker>/<agent>.json and returns the parsed dict.

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
            content = content.decode("utf-8")
        return json.loads(content)
    except Exception as e:
        logger.debug("[OSSReader] Failed to fetch %s: %s", key, e)
        return None


async def async_fetch_json_report(date: str, ticker: str, agent: str = "portfolio_manager") -> Optional[Dict[str, Any]]:
    """Async wrapper for fetch_json_report."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fetch_json_report, date, ticker, agent)
