#!/usr/bin/env python3
"""Test script for the Bear Researcher Agent."""

import asyncio
import os
from datetime import datetime

from agentscope.model import OpenAIChatModel

from tradingscope.agents.researchers.bear_researcher import create_bear_researcher_agent

async def main():
    # Initialize the model (using a mock for testing)
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
    from tradingscope.agents.utils.context import AgentContext
    context = AgentContext()
    context.company_of_interest = "AAPL"
    context.market_report = market_report
    context.sentiment_report = sentiment_report
    context.news_report = news_report
    context.fundamentals_report = fundamentals_report
    context.trade_date = trade_date

    # Create the bear researcher agent
    agent = create_bear_researcher_agent(
        model=model,
        context=context,
    )

    print("🐻 Bear Researcher Agent created successfully!")
    print(f"Agent name: {agent.name}")

    await agent(None)

if __name__ == "__main__":
    asyncio.run(main())
