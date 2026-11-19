"""Shared models, dates, and typed workflow state for TradingScope agents."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Annotated

import yfinance as yf
from agentscope.credential import OpenAICredential
from yfinance.exceptions import YFRateLimitError

from tradingscope.agents.output import (
    FundamentalsAnalystOutput,
    MarketAnalystOutput,
    NewsAnalystOutput,
    PortfolioManagerOutput,
    ResearchManagerOutput,
    SocialMediaAnalystOutput,
    TraderOutput,
)
from tradingscope.agents.renderers import render_evidence_pack
from tradingscope.agents.utils.cache_usage import CacheUsageCollector, CacheUsageMiddleware
from tradingscope.agents.utils.dashscope_response_model import DashScopeResponseModel
from tradingscope.agents.utils.prompt_cache import CACHE_PREFIX_BREAK, DashScopeResponseFormatter
from tradingscope.agents.utils.tracing import create_tracing_middlewares
from tradingscope.dataflows.stockstats_utils import completed_daily_cutoff
from tradingscope.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def get_last_completed_us_session() -> str:
    """Return the most recent completed US trading session date."""
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="5d")
        if hist is not None and not hist.empty:
            try:
                metadata = spy.history_metadata
            except Exception:
                logger.warning("SPY schedule unavailable; using conservative New York close")
                metadata = {}
            cutoff = completed_daily_cutoff(metadata, hist.index.tz, us_market=True)
            hist = hist[hist.index.date <= cutoff]
            if not hist.empty:
                result = hist.index[-1].strftime("%Y-%m-%d")
                logger.info("Last completed US session from Yahoo Finance: %s", result)
                return result
    except YFRateLimitError:
        logger.warning("Yahoo Finance rate limited when fetching last completed session")
    except Exception as exc:
        logger.warning("Failed to get last completed session from Yahoo Finance: %s", exc)

    raise RuntimeError("Cannot determine last completed US session from Yahoo Finance history")


class AgentContext:
    """Shared workflow state whose stage outputs are validated models."""

    ticker: Annotated[str, "Ticker being analyzed"] = ""
    analysis_date: Annotated[str, "Analysis date"] = ""
    last_completed_session: Annotated[str, "Last completed US trading session"] = ""

    def __init__(self) -> None:
        self.cache_usage = CacheUsageCollector()
        self.middlewares = [CacheUsageMiddleware(self.cache_usage), *create_tracing_middlewares()]
        self.analysis_date = datetime.now().strftime("%Y-%m-%d")
        self.last_completed_session = get_last_completed_us_session()
        self.market: MarketAnalystOutput | None = None
        self.fundamentals: FundamentalsAnalystOutput | None = None
        self.news: NewsAnalystOutput | None = None
        self.social_media: SocialMediaAnalystOutput | None = None
        self.research_manager: ResearchManagerOutput | None = None
        self.trader: TraderOutput | None = None
        self.portfolio_manager: PortfolioManagerOutput | None = None

        credential = OpenAICredential(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url=DEFAULT_CONFIG["backend_url"],
        )
        formatter = DashScopeResponseFormatter()
        client_kwargs = {
            "default_headers": {
                "x-dashscope-session-cache": "enable",
            },
        }
        self.model = DashScopeResponseModel(
            credential=credential,
            model=DEFAULT_CONFIG["deep_think_llm"],
            parameters=DashScopeResponseModel.Parameters(
                thinking_enable=True,
            ),
            stream=True,
            formatter=formatter,
            client_kwargs=client_kwargs,
        )
        self.non_thinking_model = DashScopeResponseModel(
            credential=credential,
            model=DEFAULT_CONFIG["deep_think_llm"],
            parameters=DashScopeResponseModel.Parameters(
                thinking_enable=False,
            ),
            stream=True,
            formatter=formatter,
            client_kwargs=client_kwargs,
        )

    def generate_analyst_reports_md(self) -> str:
        """Render compact analyst evidence packs shared by every downstream stage."""
        outputs = (
            self.market,
            self.social_media,
            self.news,
            self.fundamentals,
        )
        return "\n\n---\n\n".join(render_evidence_pack(output) for output in outputs if output is not None)

    def generate_trader_context_md(self) -> str:
        """Append the research pack so the analyst layer remains a cacheable prefix."""
        research = render_evidence_pack(self.research_manager) if self.research_manager is not None else ""
        return f"{self.generate_analyst_reports_md()}{CACHE_PREFIX_BREAK}{research}"

    def generate_risk_evaluation_context_md(self) -> str:
        """Append the trader pack so analyst and research layers remain cacheable prefixes."""
        trader = render_evidence_pack(self.trader) if self.trader is not None else ""
        return f"{self.generate_trader_context_md()}{CACHE_PREFIX_BREAK}{trader}"
