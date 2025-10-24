#!/usr/bin/env python3
"""Example script demonstrating how to use the Social Media Analyst Agent."""

import asyncio
import os
from datetime import datetime

from agentscope.model import OpenAIChatModel

from tradingscope.agents.analysts.social_media_analyst import create_social_media_analyst_agent


async def main():
    model = OpenAIChatModel(
        model_name="qwen-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
    )
    trade_date = datetime.now().strftime("%Y-%m-%d")

    # Create AgentContext
    from tradingscope.agents.utils.context import AgentContext
    context = AgentContext()
    context.company_of_interest = "AAPL"
    context.trade_date = trade_date

    agent = create_social_media_analyst_agent(model=model, context=context)
    await agent(None)


if __name__ == "__main__":
    asyncio.run(main())
