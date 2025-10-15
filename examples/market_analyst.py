#!/usr/bin/env python3
"""Example script demonstrating how to use the Market Analyst Agent."""

import asyncio
import os

from agentscope.model import OpenAIChatModel

from tradingscope.agents.analysts.market_analyst import create_market_analyst_agent


async def main():
    model = OpenAIChatModel(
        model_name="qwen-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
    )

    #trade_date = datetime.now().strftime("%Y-%m-%d")
    trade_date = "2025-10-14"

    agent = create_market_analyst_agent(model=model, ticker="BABA", trade_date=trade_date)
    await agent(None)


if __name__ == "__main__":
    asyncio.run(main())
