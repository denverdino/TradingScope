#!/usr/bin/env python3
"""Example script demonstrating how to use the Fundamentals Analyst Agent."""

import asyncio
import os
from datetime import datetime

from agentscope.model import OpenAIChatModel

from tradingscope.agents.analysts.fundamentals_analyst import create_fundamentals_analyst_agent


async def main():
    model = OpenAIChatModel(
        model_name="qwen-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
    )

    current_date = datetime.now().strftime("%Y-%m-%d")

    agent = create_fundamentals_analyst_agent(model=model, current_date=current_date, ticker="AAPL")
    await agent(None)


asyncio.run(main())
