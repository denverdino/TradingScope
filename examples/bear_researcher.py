#!/usr/bin/env python3
"""Test script for the Bear Researcher Agent with long-term memory."""

import asyncio

from tradingscope.agents.researchers.bear_researcher import create_bear_researcher_agent
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.memory_manager import FinancialMemoryManager


async def main():
    # Create AgentContext (trade_date defaults to today)
    context = AgentContext()
    context.company_of_interest = "AAPL"
    context.market_report = "Market report content here"
    context.sentiment_report = "Sentiment report content here"
    context.news_report = "News report content here"
    context.fundamentals_report = "Fundamentals report content here"

    # Create memory manager for long-term memory
    memory_manager = FinancialMemoryManager()

    try:
        # Create the bear researcher agent with long-term memory
        agent = create_bear_researcher_agent(
            context=context,
            long_term_memory=memory_manager.get_readonly_memory(),
            long_term_memory_mode="static_control",
        )

        print(f"Bear Researcher Agent created: {agent.name}")

        await agent(None)
    finally:
        await memory_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
