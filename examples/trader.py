#!/usr/bin/env python3
"""Example script demonstrating how to use the Trader Agent."""

import asyncio
import os
from datetime import datetime

from agentscope.model import OpenAIChatModel

from tradingscope.agents.trader.trader import create_trader_agent


async def main():
    # Initialize the model
    model = OpenAIChatModel(
        model_name="qwen-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
    )

    # Create sample data for testing
    company_of_interest = "AAPL"
    investment_plan = "Based on technical analysis, consider entering a long position with a target price of $200."
    market_research_report = "Technical indicators show a bullish trend with strong support at $180."
    sentiment_report = "Social media sentiment is positive with increasing mentions of the company."
    news_report = "Recent news indicates strong quarterly earnings and product launches."
    fundamentals_report = "The company has strong fundamentals with a P/E ratio of 25 and consistent revenue growth."
    trade_date = datetime.now().strftime("%Y-%m-%d")

    # Create the trader agent
    agent = create_trader_agent(
        model=model,
        company_of_interest=company_of_interest,
        investment_plan=investment_plan,
        market_research_report=market_research_report,
        sentiment_report=sentiment_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        trade_date=trade_date,
    )

    # Run the agent
    response = await agent(None)
    print(f"Trader Response: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
