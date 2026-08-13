"""End-to-end workflow using strict schema-v2 agent outputs."""

from __future__ import annotations

import asyncio
from datetime import date

from agentscope import logger

from tradingscope.agents.managers.portfolio_manager import create_portfolio_manager_agent
from tradingscope.utils.oss_structured_output_uploader import (
    analysis_persistence_session,
    persist_analysis_result,
    persist_node_output,
)

from .analysts.fundamentals_analyst import create_fundamentals_analyst_agent
from .analysts.market_analyst import create_market_analyst_agent
from .analysts.news_analyst import create_news_analyst_agent
from .analysts.social_media_analyst import create_social_media_analyst_agent
from .managers.research_manager import create_research_manager_agent
from .output import (
    AnalysisResult,
    AnalystOutputs,
    FundamentalsAnalystOutput,
    MarketAnalystOutput,
    NewsAnalystOutput,
    SocialMediaAnalystOutput,
    TraderOutput,
)
from .researchers.bear_researcher import create_bear_researcher_agent
from .researchers.bull_researcher import create_bull_researcher_agent
from .researchers.debate_orchestrator import create_research_debate_orchestrator
from .risk_mgmt.aggressive_debator import create_aggressive_debator_agent
from .risk_mgmt.conservative_debator import create_conservative_debator_agent
from .risk_mgmt.debate_orchestrator import create_debate_orchestrator
from .risk_mgmt.neutral_debator import create_neutral_debator_agent
from .trader.trader import create_trader_agent
from .utils.context import AgentContext
from .utils.structured_output import StructuredAgentRunner


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


async def analyze(ticker: str, trade_date: str | None = None) -> AnalysisResult:
    """Run the seven-stage workflow and return one validated result object.

    All seven JSON-producing nodes pass through ``StructuredAgentRunner``.
    Validation failures propagate and stop downstream stages.
    """
    context = AgentContext()
    context.company_of_interest = ticker
    if trade_date:
        context.trade_date = trade_date
    workflow_trade_date = _as_date(context.trade_date).isoformat()
    async with analysis_persistence_session(workflow_trade_date, ticker):
        return await _run_analysis(ticker, context, workflow_trade_date)


async def _run_analysis(ticker: str, context: AgentContext, workflow_trade_date: str) -> AnalysisResult:
    """Execute one initialized and persistence-guarded workflow run."""

    structured_runner = StructuredAgentRunner(
        context.non_thinking_model,
        usage_collector=context.cache_usage,
    )

    market_analyst = create_market_analyst_agent(context=context)
    fundamentals_analyst = create_fundamentals_analyst_agent(context=context)
    news_analyst = create_news_analyst_agent(context=context)
    social_media_analyst = create_social_media_analyst_agent(context=context)

    async def run_analyst(agent_name, agent, output_model):
        output = await structured_runner.run(agent, output_model)
        await persist_node_output(
            agent_name,
            output,
            ticker=ticker,
            trade_date=workflow_trade_date,
        )
        return output

    analyst_tasks = [
        asyncio.create_task(
            run_analyst("market_analyst", market_analyst, MarketAnalystOutput),
        ),
        asyncio.create_task(
            run_analyst(
                "fundamentals_analyst",
                fundamentals_analyst,
                FundamentalsAnalystOutput,
            ),
        ),
        asyncio.create_task(
            run_analyst("news_analyst", news_analyst, NewsAnalystOutput),
        ),
        asyncio.create_task(
            run_analyst(
                "social_media_analyst",
                social_media_analyst,
                SocialMediaAnalystOutput,
            ),
        ),
    ]
    try:
        market, fundamentals, news, social_media = await asyncio.gather(*analyst_tasks)
    except BaseException:
        for task in analyst_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*analyst_tasks, return_exceptions=True)
        raise
    context.market_analysis = market
    context.fundamentals_analysis = fundamentals
    context.news_analysis = news
    context.social_analysis = social_media
    logger.info("分析师结构化输出已验证，开始研究辩论")

    research_orchestrator = create_research_debate_orchestrator(
        bull_researcher=create_bull_researcher_agent(context=context),
        bear_researcher=create_bear_researcher_agent(context=context),
        research_manager=create_research_manager_agent(context=context),
        structured_runner=structured_runner,
        max_rounds=2,
        reference_outputs=(market, fundamentals, news, social_media),
    )
    research_manager = await research_orchestrator.run_debate(company_name=ticker)
    context.research_decision = research_manager
    await persist_node_output(
        "research_manager",
        research_manager,
        ticker=ticker,
        trade_date=workflow_trade_date,
    )

    trader_agent = create_trader_agent(context=context)
    trader = await structured_runner.run(
        trader_agent,
        TraderOutput,
        reference_outputs=(research_manager, market, fundamentals, news, social_media),
    )
    context.trader_decision = trader
    await persist_node_output(
        "trader",
        trader,
        ticker=ticker,
        trade_date=workflow_trade_date,
    )

    risk_orchestrator = create_debate_orchestrator(
        aggressive_agent=create_aggressive_debator_agent(context=context),
        conservative_agent=create_conservative_debator_agent(context=context),
        neutral_agent=create_neutral_debator_agent(context=context),
        portfolio_manager=create_portfolio_manager_agent(context=context),
        structured_runner=structured_runner,
        max_rounds=2,
        reference_outputs=(trader, research_manager, market, fundamentals, news, social_media),
    )
    portfolio_manager = await risk_orchestrator.run_debate(company_name=ticker)
    context.portfolio_decision = portfolio_manager
    await persist_node_output(
        "portfolio_manager",
        portfolio_manager,
        ticker=ticker,
        trade_date=workflow_trade_date,
    )

    result = AnalysisResult(
        schema_version="2.0",
        ticker=ticker,
        trade_date=_as_date(context.trade_date),
        latest_trading_date=_as_date(context.latest_trading_date),
        analysts=AnalystOutputs(
            market=market,
            fundamentals=fundamentals,
            news=news,
            social_media=social_media,
        ),
        research_manager=research_manager,
        trader=trader,
        portfolio_manager=portfolio_manager,
    )
    context.cache_usage.log_summary()
    await persist_analysis_result(result)
    return result
