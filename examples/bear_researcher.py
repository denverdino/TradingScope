#!/usr/bin/env python3
"""Example script for the Bear Researcher Agent."""

import asyncio

from tradingscope.agents.researchers.bear_researcher import create_bear_researcher_agent
from tradingscope.agents.utils.context import AgentContext


async def main():
    # Create AgentContext (trade_date defaults to today)
    context = AgentContext()
    context.company_of_interest = "AAPL"
    context.market_report = "Market report content here"
    context.sentiment_report = "Sentiment report content here"
    context.news_report = "News report content here"
    context.fundamentals_report = "Fundamentals report content here"

    # Create the bear researcher agent
    agent = create_bear_researcher_agent(
        context=context,
    )

    print(f"Bear Researcher Agent created: {agent.name}")

    await agent(None)


if __name__ == "__main__":
    asyncio.run(main())
