#!/usr/bin/env python3
"""Example script demonstrating the research multi-agent debate system."""

import asyncio

from tradingscope.agents.managers.research_manager import create_research_manager_agent
from tradingscope.agents.researchers.bear_researcher import create_bear_researcher_agent
from tradingscope.agents.researchers.bull_researcher import create_bull_researcher_agent
from tradingscope.agents.researchers.debate_orchestrator import create_research_debate_orchestrator
from tradingscope.agents.utils.context import AgentContext


async def main():
    """Example usage of the research debate system."""

    context = AgentContext()
    context.company_of_interest = "AAPL"
    context.market_report = "Technical indicators show a bullish trend with strong support at $180."
    context.sentiment_report = "Social media sentiment is positive with increasing mentions of the company."
    context.news_report = "Recent news indicates strong quarterly earnings and product launches."
    context.fundamentals_report = "The company has strong fundamentals with a P/E ratio of 25 and consistent revenue growth."

    bull_researcher = create_bull_researcher_agent(context=context)
    bear_researcher = create_bear_researcher_agent(context=context)
    research_manager = create_research_manager_agent(context=context)

    orchestrator = create_research_debate_orchestrator(
        bull_researcher=bull_researcher,
        bear_researcher=bear_researcher,
        research_manager=research_manager,
        max_rounds=3,
    )

    print(f"Starting research debate for {context.company_of_interest}")
    print("=" * 60)

    final_decision = await orchestrator.run_debate(
        company_name=context.company_of_interest,
    )

    print("=" * 60)
    print("Final Decision from Research Manager:")
    print(final_decision.get_text_content())
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
