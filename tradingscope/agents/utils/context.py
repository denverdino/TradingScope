"""Shared models, dates, and typed workflow state for TradingScope agents."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Annotated, Any

import yfinance as yf
from agentscope.credential import DashScopeCredential
from agentscope.formatter import DashScopeChatFormatter
from agentscope.model import DashScopeChatModel
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
from tradingscope.agents.renderers import render_markdown
from tradingscope.default_config import DEFAULT_CONFIG


class CodeInterpreterModel(DashScopeChatModel):
    """DashScope model with its server-side code interpreter enabled."""

    async def _call_api(
        self,
        model_name: str,
        messages: list,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ):
        kwargs.setdefault("extra_body", {})
        kwargs["extra_body"]["enable_code_interpreter"] = True
        return await super()._call_api(
            model_name,
            messages,
            tools=None,
            tool_choice=None,
            **kwargs,
        )


logger = logging.getLogger(__name__)


def get_latest_us_trading_date() -> str:
    """Return the most recent completed US trading date."""
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="5d")
        if hist is not None and not hist.empty:
            latest_date = hist.index[-1]
            if hasattr(latest_date, "tz") and latest_date.tz is not None:
                latest_date = latest_date.tz_localize(None)
            result = latest_date.strftime("%Y-%m-%d")
            logger.info("Latest US trading date from Yahoo Finance: %s", result)
            return result
    except YFRateLimitError:
        logger.warning("Yahoo Finance rate limited when fetching latest trading date")
    except Exception as exc:
        logger.warning("Failed to get latest trading date from Yahoo Finance: %s", exc)

    now = datetime.now()
    if now.weekday() == 5:
        now -= timedelta(days=1)
    elif now.weekday() == 6:
        now -= timedelta(days=2)
    result = now.strftime("%Y-%m-%d")
    logger.info("Using fallback trading date (weekday-based): %s", result)
    return result


class AgentContext:
    """Shared workflow state whose analysis values are validated models."""

    company_of_interest: Annotated[str, "Company being analyzed"] = ""
    trade_date: Annotated[str, "Current analysis date"] = ""
    latest_trading_date: Annotated[str, "Latest US trading date"] = ""

    def __init__(self) -> None:
        self.trade_date = datetime.now().strftime("%Y-%m-%d")
        self.latest_trading_date = get_latest_us_trading_date()
        self.market_analysis: MarketAnalystOutput | None = None
        self.fundamentals_analysis: FundamentalsAnalystOutput | None = None
        self.news_analysis: NewsAnalystOutput | None = None
        self.social_analysis: SocialMediaAnalystOutput | None = None
        self.research_decision: ResearchManagerOutput | None = None
        self.trader_decision: TraderOutput | None = None
        self.portfolio_decision: PortfolioManagerOutput | None = None

        credential = DashScopeCredential(api_key=os.environ.get("DASHSCOPE_API_KEY"))
        common_parameters = DashScopeChatModel.Parameters(
            thinking_enable=True,
            parallel_tool_calls=False,
        )
        self.model = DashScopeChatModel(
            credential=credential,
            model=DEFAULT_CONFIG["deep_think_llm"],
            parameters=common_parameters,
            stream=True,
            formatter=DashScopeChatFormatter(),
        )
        self.non_thinking_model = DashScopeChatModel(
            credential=credential,
            model=DEFAULT_CONFIG["deep_think_llm"],
            parameters=DashScopeChatModel.Parameters(
                thinking_enable=False,
                parallel_tool_calls=False,
            ),
            stream=True,
            formatter=DashScopeChatFormatter(),
        )
        self.code_interpreter_model = CodeInterpreterModel(
            credential=credential,
            model=DEFAULT_CONFIG["deep_think_llm"],
            parameters=CodeInterpreterModel.Parameters(
                thinking_enable=True,
                parallel_tool_calls=False,
            ),
            stream=True,
            formatter=DashScopeChatFormatter(),
        )

    def generate_analyst_reports_md(self) -> str:
        """Render available typed analyst outputs for downstream text agents."""
        outputs = (
            self.market_analysis,
            self.social_analysis,
            self.news_analysis,
            self.fundamentals_analysis,
        )
        return "\n\n---\n\n".join(render_markdown(output) for output in outputs if output is not None)

    def generate_trader_context_md(self) -> str:
        """Render the validated research decision and analyst evidence."""
        research = render_markdown(self.research_decision) if self.research_decision is not None else ""
        return f"""## 研究经理投资建议

{research}

---

{self.generate_analyst_reports_md()}"""

    def generate_risk_evaluation_context_md(self) -> str:
        """Render the validated trader decision and its upstream context."""
        trader = render_markdown(self.trader_decision) if self.trader_decision is not None else ""
        return f"""## 交易员操作计划

{trader}

---

{self.generate_trader_context_md()}"""
