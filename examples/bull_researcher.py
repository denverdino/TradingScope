#!/usr/bin/env python3
"""Example script for the Bull Researcher Agent."""

import asyncio

from tradingscope.agents.researchers.bull_researcher import create_bull_researcher_agent
from tradingscope.agents.utils.agent_utils import call_agent_with_retry
from tradingscope.agents.utils.context import AgentContext


async def main():
    context = AgentContext()
    context.company_of_interest = "AAPL"
    context.market_report = "Market report content here"
    context.sentiment_report = "Sentiment report content here"
    context.news_report = "News report content here"
    context.fundamentals_report = "Fundamentals report content here"

    agent = create_bull_researcher_agent(context=context)

    print(f"Bull Researcher Agent created: {agent.name}")

    response = await call_agent_with_retry(agent, None)
    print(response.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())
