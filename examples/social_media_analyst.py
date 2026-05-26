#!/usr/bin/env python3
"""Example script demonstrating how to use the Social Media Analyst Agent."""

import asyncio

from tradingscope.agents.analysts.social_media_analyst import create_social_media_analyst_agent
from tradingscope.agents.utils.agent_utils import call_agent_with_retry
from tradingscope.agents.utils.context import AgentContext


async def main():
    context = AgentContext()
    context.company_of_interest = "MSFT"

    agent = create_social_media_analyst_agent(context=context)
    response = await call_agent_with_retry(agent, None)

    print("=" * 60)
    print("Social Media Analyst Response:")
    print(response.get_text_content())
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
