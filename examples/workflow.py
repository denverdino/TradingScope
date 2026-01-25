#!/usr/bin/env python3
"""Test script for the workflow"""

import asyncio
import os
from datetime import datetime

from agentscope.model import OpenAIChatModel

from tradingscope.agents.workflow import analyze

model = OpenAIChatModel(
    model_name="qwen-plus",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    stream=True,
)

trade_date = datetime.now().strftime("%Y-%m-%d")


async def main():
    report = await analyze(model, "AAPL", trade_date)
    print(f"Final Report {trade_date}")
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
