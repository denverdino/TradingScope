#!/usr/bin/env python3
"""Example script for the Equity Analyst Agent (Qwen Deep Research)."""

import asyncio

from tradingscope.agents.analysts.equity_analyst import (
    create_equity_analyst_agent,
    run_equity_analysis,
)
from tradingscope.agents.utils.context import AgentContext


async def main():
    context = AgentContext()
    context.company_of_interest = "BABA"

    agent = create_equity_analyst_agent(context=context)
    result = await run_equity_analysis(agent)

    print("Equity Analyst Report:")
    print(result.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())
