#!/usr/bin/env python3
"""Example script demonstrating how to use the Trader Agent."""

import asyncio

from tradingscope.agents.trader.trader import create_trader_agent
from tradingscope.agents.utils.agent_utils import call_agent_with_retry
from tradingscope.agents.utils.context import AgentContext


async def main():
    context = AgentContext()
    context.company_of_interest = "AAPL"
    context.market_report = "Technical indicators show a bullish trend with strong support at $180."
    context.sentiment_report = "Social media sentiment is positive with increasing mentions of the company."
    context.news_report = "Recent news indicates strong quarterly earnings and product launches."
    context.fundamentals_report = "The company has strong fundamentals with a P/E ratio of 25 and consistent revenue growth."
    context.researcher_investment_plan = "Based on technical analysis, consider entering a long position with a target price of $200."

    agent = create_trader_agent(context=context)
    response = await call_agent_with_retry(agent, None)

    print("=" * 60)
    print("Trader Response:")
    print(response.get_text_content())
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
