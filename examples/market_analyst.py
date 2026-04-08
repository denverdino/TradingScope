#!/usr/bin/env python3
"""Example script demonstrating how to use the Market Analyst Agent."""

import asyncio

from tradingscope.agents.analysts.market_analyst import create_market_analyst_agent
from tradingscope.agents.utils.context import AgentContext


async def main():
    context = AgentContext()
    context.company_of_interest = "BABA"

    agent = create_market_analyst_agent(context=context)
    await agent(None)


if __name__ == "__main__":
    asyncio.run(main())
