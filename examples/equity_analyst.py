#!/usr/bin/env python3
"""Example script for the Equity Analyst Agent (Qwen Deep Research)."""

import asyncio

from tradingscope.agents.analysts.equity_analyst import create_equity_analyst_agent
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.workflow import get_content, get_markdown


async def main():
    context = AgentContext()
    context.company_of_interest = "BABA"

    agent = create_equity_analyst_agent(context=context)
    result = await agent.analyze()

    print("Equity Analyst Report:")
    print(get_markdown(get_content(result), 2))


if __name__ == "__main__":
    asyncio.run(main())
