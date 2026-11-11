"""Read agent structured outputs (JSON) from Alibaba Cloud OSS.

Discovers and fetches structured outputs stored at: tradingscope/<date>/<ticker>/<agent_name>.json

Requires OSS SDK configuration:
    OSS_REGION, OSS_BUCKET, OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date as date_type
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

_OSS_KEY_PREFIX = "tradingscope"


class CompletionManifest(BaseModel):
    """Strict marker proving that every schema-v3 artifact was written."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["3.0"]
    status: Literal["complete"]
    ticker: str
    analysis_date: date_type


# Remove migration once no v2 data remains in the fourteen-day evaluation window.
def _upgrade_v2(data: Any) -> Any:
    """Recursively copy schema-v2 data using schema-v3 field names and versions."""
    if isinstance(data, dict):
        renamed_keys = {
            "trade_date": "analysis_date",
            "latest_trading_date": "last_completed_session",
            "entry_price": "execution_price",
            "entry_price_low": "execution_price_low",
            "entry_price_high": "execution_price_high",
        }
        if data.keys() & renamed_keys.values():
            raise ValueError("schema-v2 data must use historical field names")
        return {renamed_keys.get(key, key): "3.0" if key == "schema_version" and value == "2.0" else _upgrade_v2(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_upgrade_v2(item) for item in data]
    return data


def _get_client_and_bucket():
    """Get OSS client and bucket name from the uploader module."""
    import tradingscope.utils.oss_structured_output_uploader as uploader

    client = uploader._get_client()
    bucket_name = uploader._get_bucket_name()
    return client, bucket_name


def fetch_structured_output(date: str, ticker: str, agent: str = "portfolio_manager") -> Optional[Dict[str, Any]]:
    """Fetch a single structured output's content from OSS via SDK.

    Args:
        date: Analysis date (YYYY-MM-DD)
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


def fetch_completed_output(
    date: str,
    ticker: str,
    agent: str = "portfolio_manager",
) -> Optional[Dict[str, Any]]:
    """Fetch completed schema-v3 data, upgrading historical v2 artifacts."""
    manifest_data = fetch_structured_output(date, ticker, "manifest")
    try:
        if isinstance(manifest_data, dict) and manifest_data.get("schema_version") == "2.0":
            manifest_data = _upgrade_v2(manifest_data)
        manifest = CompletionManifest.model_validate(manifest_data)
    except ValueError:
        return None
    if manifest.ticker != ticker or manifest.analysis_date.isoformat() != date:
        return None

    output = fetch_structured_output(date, ticker, agent)
    if isinstance(output, dict) and output.get("schema_version") == "2.0":
        try:
            output = _upgrade_v2(output)
        except ValueError:
            return None
    if not isinstance(output, dict) or output.get("schema_version") != "3.0":
        return None
    return output


async def async_fetch_completed_output(
    date: str,
    ticker: str,
    agent: str = "portfolio_manager",
) -> Optional[Dict[str, Any]]:
    """Async wrapper for manifest-gated schema-v3 reads with v2 migration."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        fetch_completed_output,
        date,
        ticker,
        agent,
    )
