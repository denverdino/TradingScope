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

if __name__ == "__main__":
    asyncio.run(analyze(model, "AAPL", trade_date))
