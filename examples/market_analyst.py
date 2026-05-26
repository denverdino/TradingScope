#!/usr/bin/env python3
"""Example script demonstrating how to use the Market Analyst Agent."""

import asyncio

from tradingscope.agents.analysts.market_analyst import create_market_analyst_agent
from tradingscope.agents.utils.agent_utils import call_agent_with_retry
from tradingscope.agents.utils.context import AgentContext


async def main():
    context = AgentContext()
    context.company_of_interest = "NVDA"

    agent = create_market_analyst_agent(context=context)
    response = await call_agent_with_retry(agent, None)

    print("=" * 60)
    print("Market Analyst Response:")
    print(response.get_text_content())
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
