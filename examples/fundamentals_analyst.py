#!/usr/bin/env python3
"""Example script demonstrating how to use the Fundamentals Analyst Agent."""

import asyncio
import os
from datetime import datetime

from agentscope.model import OpenAIChatModel

from tradingscope.agents.analysts.fundamentals_analyst import create_fundamentals_analyst_agent
from tradingscope.default_config import DEFAULT_CONFIG


async def main():
    model = OpenAIChatModel(
        model_name=DEFAULT_CONFIG["deep_think_llm"],
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
    )

    current_date = datetime.now().strftime("%Y-%m-%d")

    # Create AgentContext
    from tradingscope.agents.utils.context import AgentContext
    context = AgentContext()
    context.company_of_interest = "MSFT"
    context.trade_date = current_date

    agent = create_fundamentals_analyst_agent(model=model, context=context)
    await agent(None)


asyncio.run(main())
