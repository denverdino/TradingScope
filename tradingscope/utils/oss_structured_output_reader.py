"""Read current completed analysis bundles from OSS or local storage."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date as date_type
from datetime import timedelta
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from tradingscope.agents.output import AgentOutputBase, AnalysisResult, PortfolioManagerOutput, TraderOutput
from tradingscope.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

_OSS_KEY_PREFIX = "tradingscope"

STAGE_NAMES = (
    "market_analyst",
    "fundamentals_analyst",
    "news_analyst",
    "social_media_analyst",
    "research_manager",
    "trader",
    "portfolio_manager",
)


def is_safe_path_component(value: str) -> bool:
    return value not in {".", ".."} and not any(character in value for character in ("/", chr(92), chr(0)))


class CompletionManifest(BaseModel):
    """Completion marker for a current immutable analysis bundle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["3.1"]
    status: Literal["complete"]
    ticker: str = Field(min_length=1)
    analysis_date: date_type
    run_id: str = Field(min_length=1)
    code_version: str | None
    models: dict[str, str]
    last_completed_session: date_type
    forecast_started_at: AwareDatetime
    forecast_completed_at: AwareDatetime

    @field_validator("run_id", "ticker")
    @classmethod
    def safe_path_component(cls, value: str) -> str:
        if not is_safe_path_component(value):
            raise ValueError("identity must be a safe path component")
        return value

    @model_validator(mode="after")
    def validate_forecast_timestamps(self) -> CompletionManifest:
        if self.forecast_completed_at < self.forecast_started_at:
            raise ValueError("forecast cannot finish before it started")
        return self


@dataclass(frozen=True)
class CompletedRun:
    """The validated stage outputs of one completed analysis run."""

    manifest: CompletionManifest
    outputs: dict[str, AgentOutputBase]

    @property
    def run_id(self) -> str:
        return self.manifest.run_id


def _get_client_and_bucket():
    from tradingscope.utils import oss_structured_output_uploader as uploader

    return uploader._get_client(), uploader._get_bucket_name()


def fetch_structured_output(date: str, ticker: str, agent: str = "portfolio_manager", *, results_dir: str | None = None) -> dict[str, Any] | None:
    """Read one raw JSON artifact without asserting that it is a completed run."""
    client, bucket_name = _get_client_and_bucket()
    if client is None:
        root = (Path(results_dir or DEFAULT_CONFIG["results_dir"]) / "data").resolve()
        path = (root / date / ticker / f"{agent}.json").resolve()
        if not path.is_relative_to(root):
            raise ValueError("artifact path must remain within the results directory")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    import alibabacloud_oss_v2 as oss

    key = f"{_OSS_KEY_PREFIX}/{date}/{ticker}/{agent}.json"
    try:
        response = client.get_object(oss.GetObjectRequest(bucket=bucket_name, key=key))
        return json.loads(response.body.content)
    except Exception:
        logger.debug("[OSSStructuredReader] Could not read %s", key)
        return None


async def async_fetch_structured_output(date: str, ticker: str, agent: str = "portfolio_manager") -> dict[str, Any] | None:
    return await asyncio.to_thread(fetch_structured_output, date, ticker, agent)


def _validate_manifest(data: Any, date: str, ticker: str) -> CompletionManifest | None:
    try:
        manifest = CompletionManifest.model_validate(data)
    except ValidationError:
        return None
    if manifest.ticker != ticker or manifest.analysis_date.isoformat() != date:
        return None
    return manifest


def fetch_completed_manifest(date: str, ticker: str, *, results_dir: str | None = None) -> CompletionManifest | None:
    read = partial(fetch_structured_output, results_dir=results_dir) if results_dir is not None else fetch_structured_output
    return _validate_manifest(read(date, ticker, "manifest"), date, ticker)


def fetch_completed_output(date: str, ticker: str, agent: str = "portfolio_manager") -> dict[str, Any] | None:
    """Read a stage only from a validated current completed bundle."""
    run = fetch_completed_run(date, ticker)
    if run is None:
        return None
    if agent == "full_report":
        return fetch_structured_output(date, ticker, f"runs/{run.run_id}/full_report")
    output = run.outputs.get(agent)
    return output.model_dump(mode="json") if output is not None else None


async def async_fetch_completed_output(date: str, ticker: str, agent: str = "portfolio_manager") -> dict[str, Any] | None:
    return await asyncio.to_thread(fetch_completed_output, date, ticker, agent)


def fetch_completed_run(date: str, ticker: str, *, results_dir: str | None = None) -> CompletedRun | None:
    """Validate a current manifest, its complete bundle, and all stage identities."""
    read = partial(fetch_structured_output, results_dir=results_dir) if results_dir is not None else fetch_structured_output
    manifest = _validate_manifest(read(date, ticker, "manifest"), date, ticker)
    if manifest is None:
        return None
    data = read(date, ticker, f"runs/{manifest.run_id}/full_report")
    try:
        result = AnalysisResult.model_validate(data)
    except ValidationError:
        return None
    if result.ticker != ticker or result.analysis_date != manifest.analysis_date or result.last_completed_session != manifest.last_completed_session:
        return None
    outputs = {
        "market_analyst": result.analysts.market,
        "fundamentals_analyst": result.analysts.fundamentals,
        "news_analyst": result.analysts.news,
        "social_media_analyst": result.analysts.social_media,
        "research_manager": result.research_manager,
        "trader": result.trader,
        "portfolio_manager": result.portfolio_manager,
    }
    for name, output in outputs.items():
        if output.ticker != ticker or output.analysis_date != manifest.analysis_date or output.agent_name != name:
            return None
        if output.last_completed_session != manifest.last_completed_session:
            return None
        if isinstance(output, (TraderOutput, PortfolioManagerOutput)) and output.trade_intent is None:
            return None
    if _validate_manifest(read(date, ticker, "manifest"), date, ticker) != manifest:
        return None
    return CompletedRun(manifest=manifest, outputs=outputs)


def latest_completed_session(ticker: str, through: date_type, lookback_days: int) -> date_type | None:
    dates = [(through - timedelta(days=offset)).isoformat() for offset in range(lookback_days + 1)]
    _get_client_and_bucket()
    with ThreadPoolExecutor(max_workers=4) as executor:
        manifests = executor.map(fetch_completed_manifest, dates, repeat(ticker))
        return max((manifest.last_completed_session for manifest in manifests if manifest is not None), default=None)
