#!/usr/bin/env python3
"""Example script demonstrating how to use the News Analyst Agent."""

import asyncio
import os
from datetime import datetime

from agentscope.model import OpenAIChatModel

from tradingscope.agents.analysts.news_analyst import create_news_analyst_agent


async def main():
    model = OpenAIChatModel(
        model_name="qwen-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
    )
    trade_date = datetime.now().strftime("%Y-%m-%d")
    agent = create_news_analyst_agent(model=model, ticker="AAPL", trade_date=trade_date)
    await agent(None)


if __name__ == "__main__":
    asyncio.run(main())
