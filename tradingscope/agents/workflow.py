"""End-to-end workflow using strict schema-v3 agent outputs."""

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
from .risk_mgmt.aggressive_debater import create_aggressive_debater_agent
from .risk_mgmt.conservative_debater import create_conservative_debater_agent
from .risk_mgmt.debate_orchestrator import create_risk_debate_orchestrator
from .risk_mgmt.neutral_debater import create_neutral_debater_agent
from .trader.trader import create_trader_agent
from .utils.context import AgentContext
from .utils.structured_output import StructuredAgentRunner


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


async def analyze(ticker: str, analysis_date: str | None = None) -> AnalysisResult:
    """Run the seven-stage workflow and return one validated result object.

    All seven JSON-producing nodes pass through ``StructuredAgentRunner``.
    Validation failures propagate and stop downstream stages.
    """
    context = AgentContext()
    context.ticker = ticker
    if analysis_date:
        context.analysis_date = analysis_date
    workflow_analysis_date = _as_date(context.analysis_date).isoformat()
    async with analysis_persistence_session(workflow_analysis_date, ticker):
        return await _run_analysis(ticker, context, workflow_analysis_date)


async def _run_analysis(ticker: str, context: AgentContext, workflow_analysis_date: str) -> AnalysisResult:
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
            analysis_date=workflow_analysis_date,
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
    context.market = market
    context.fundamentals = fundamentals
    context.news = news
    context.social_media = social_media
    logger.info("分析师结构化输出已验证，开始研究辩论")

    research_orchestrator = create_research_debate_orchestrator(
        bull_researcher=create_bull_researcher_agent(context=context),
        bear_researcher=create_bear_researcher_agent(context=context),
        research_manager=create_research_manager_agent(context=context),
        structured_runner=structured_runner,
        max_rounds=2,
        reference_outputs=(market, fundamentals, news, social_media),
    )
    research_manager = await research_orchestrator.run_debate(ticker=ticker)
    context.research_manager = research_manager
    await persist_node_output(
        "research_manager",
        research_manager,
        ticker=ticker,
        analysis_date=workflow_analysis_date,
    )

    trader_agent = create_trader_agent(context=context)
    trader = await structured_runner.run(
        trader_agent,
        TraderOutput,
        reference_outputs=(research_manager, market, fundamentals, news, social_media),
    )
    context.trader = trader
    await persist_node_output(
        "trader",
        trader,
        ticker=ticker,
        analysis_date=workflow_analysis_date,
    )

    risk_orchestrator = create_risk_debate_orchestrator(
        aggressive_agent=create_aggressive_debater_agent(context=context),
        conservative_agent=create_conservative_debater_agent(context=context),
        neutral_agent=create_neutral_debater_agent(context=context),
        portfolio_manager=create_portfolio_manager_agent(context=context),
        structured_runner=structured_runner,
        max_rounds=2,
        reference_outputs=(trader, research_manager, market, fundamentals, news, social_media),
    )
    portfolio_manager = await risk_orchestrator.run_debate(ticker=ticker)
    context.portfolio_manager = portfolio_manager
    await persist_node_output(
        "portfolio_manager",
        portfolio_manager,
        ticker=ticker,
        analysis_date=workflow_analysis_date,
    )

    result = AnalysisResult(
        schema_version="3.0",
        ticker=ticker,
        analysis_date=_as_date(context.analysis_date),
        last_completed_session=_as_date(context.last_completed_session),
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
