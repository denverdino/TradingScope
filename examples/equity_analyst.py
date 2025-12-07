#!/usr/bin/env python3
"""Test script for the Bear Researcher Agent."""

import asyncio
import os
from datetime import datetime

from agentscope.model import OpenAIChatModel

from tradingscope.agents.analysts.equity_analyst import create_equity_analyst_agent


async def main():
    trade_date = datetime.now().strftime("%Y-%m-%d")

    # Create AgentContext
    from tradingscope.agents.utils.context import AgentContext
    context = AgentContext()
    context.company_of_interest = "BABA"
    context.trade_date = trade_date

    agent = create_equity_analyst_agent(context=context)
    result = await agent.analyze()

    print("📈 Equity Analyst Report:")
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())

