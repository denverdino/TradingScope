#!/usr/bin/env python3
"""Example script demonstrating how to use the Trader Agent with long-term memory."""

import asyncio

from tradingscope.agents.trader.trader import create_trader_agent
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.memory_manager import FinancialMemoryManager


async def main():
    # Create sample data for testing
    company_of_interest = "AAPL"
    investment_plan = "Based on technical analysis, consider entering a long position with a target price of $200."
    market_research_report = "Technical indicators show a bullish trend with strong support at $180."
    sentiment_report = "Social media sentiment is positive with increasing mentions of the company."
    news_report = "Recent news indicates strong quarterly earnings and product launches."
    fundamentals_report = "The company has strong fundamentals with a P/E ratio of 25 and consistent revenue growth."

    # Create AgentContext (trade_date defaults to today)
    context = AgentContext()
    context.company_of_interest = company_of_interest
    context.investment_plan = investment_plan
    context.market_report = market_research_report
    context.sentiment_report = sentiment_report
    context.news_report = news_report
    context.fundamentals_report = fundamentals_report

    # Create memory manager for long-term memory
    memory_manager = FinancialMemoryManager()

    try:
        # Create the trader agent with long-term memory
        agent = create_trader_agent(
            context=context,
            long_term_memory=memory_manager.get_readonly_memory(),
            long_term_memory_mode="static_control",
        )

        # Run the agent
        response = await agent(None)
        print(f"Trader Response: {response.content}")
    finally:
        await memory_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
