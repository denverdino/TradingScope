#!/usr/bin/env python3
"""Test script for the Bear Researcher Agent with long-term memory."""

import asyncio
import os
from datetime import datetime

from agentscope.model import OpenAIChatModel

from tradingscope.agents.researchers.bear_researcher import create_bear_researcher_agent
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.memory_manager import FinancialMemoryManager


async def main():
    # Initialize the model
    model = OpenAIChatModel(
        model_name="qwen-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
    )

    # Create mock reports
    market_report = "Market report content here"
    sentiment_report = "Sentiment report content here"
    news_report = "News report content here"
    fundamentals_report = "Fundamentals report content here"
    trade_date = datetime.now().strftime("%Y-%m-%d")

    # Create AgentContext
    context = AgentContext()
    context.company_of_interest = "AAPL"
    context.market_report = market_report
    context.sentiment_report = sentiment_report
    context.news_report = news_report
    context.fundamentals_report = fundamentals_report
    context.trade_date = trade_date

    # Create memory manager for long-term memory
    memory_manager = FinancialMemoryManager()

    try:
        # Create the bear researcher agent with long-term memory
        agent = create_bear_researcher_agent(
            model=model,
            context=context,
            long_term_memory=memory_manager.bear_researcher_memory,
            long_term_memory_mode="static_control",
        )

        print(f"Bear Researcher Agent created: {agent.name}")

        await agent(None)
    finally:
        await memory_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
