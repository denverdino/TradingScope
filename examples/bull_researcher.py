#!/usr/bin/env python3
"""Test script for the Bull Researcher Agent."""

import asyncio
import datetime
import os

from agentscope.model import OpenAIChatModel

from tradingscope.agents.researchers.bull_researcher import create_bull_researcher_agent


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

    # Create the bull researcher agent
    agent = create_bull_researcher_agent(
        model=model,
        company_of_interest="AAPL",
        market_research_report=market_report,
        sentiment_report=sentiment_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        trade_date=trade_date,
    )

    print("🐂 Bull Researcher Agent created successfully!")
    print(f"Agent name: {agent.name}")

    # Note: We're not actually running the agent here to avoid API calls during testing
    # In a real test, you would run: await agent(None)


if __name__ == "__main__":
    asyncio.run(main())
