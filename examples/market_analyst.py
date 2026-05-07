#!/usr/bin/env python3
"""Example script demonstrating how to use the Market Analyst Agent with structured output."""

import asyncio

from tradingscope.agents.analysts.market_analyst import create_market_analyst_agent
from tradingscope.agents.output import MarketAnalystStructuredOutput
from tradingscope.agents.utils.context import AgentContext


async def main():
    context = AgentContext()
    context.company_of_interest = "NVDA"

    agent = create_market_analyst_agent(context=context)
    response = await agent(None, structured_model=MarketAnalystStructuredOutput)

    print("=" * 60)
    print("Market Analyst Response:")
    print(response.content)
    print("=" * 60)

    structured = response.metadata if response.metadata and ("direction" in response.metadata or "action" in response.metadata) else None
    if structured:
        print("\nStructured Output:")
        model = MarketAnalystStructuredOutput.model_validate(structured)
        print(f"  Direction:    {model.direction}")
        print(f"  Action:       {model.action}")
        print(f"  Confidence:   {model.confidence}")
        print(f"  Entry Price:  {model.entry_price}")
        print(f"  Target Price: {model.target_price}")
        print(f"  Stop Loss:    {model.stop_loss}")
        print(f"  Reasoning:    {model.reasoning}")
        print(f"  Signal:       {model.signal_strength}")
        print(f"  Invalidation: {model.invalidation_conditions}")
    else:
        print("\nNo structured output available (agent did not produce structured data).")


if __name__ == "__main__":
    asyncio.run(main())
