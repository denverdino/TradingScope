#!/usr/bin/env python3
"""Example script demonstrating how to use the Fundamentals Analyst Agent with structured output."""

import asyncio

from tradingscope.agents.analysts.fundamentals_analyst import create_fundamentals_analyst_agent
from tradingscope.agents.output import FundamentalsAnalystStructuredOutput
from tradingscope.agents.utils.context import AgentContext


async def main():
    context = AgentContext()
    context.company_of_interest = "MSFT"

    agent = create_fundamentals_analyst_agent(context=context)
    response = await agent(None, structured_model=FundamentalsAnalystStructuredOutput)

    print("=" * 60)
    print("Fundamentals Analyst Response:")
    print(response.content)
    print("=" * 60)

    structured = response.metadata if response.metadata and ("direction" in response.metadata or "action" in response.metadata) else None
    if structured:
        print("\nStructured Output:")
        model = FundamentalsAnalystStructuredOutput.model_validate(structured)
        print(f"  Direction:    {model.direction}")
        print(f"  Action:       {model.action}")
        print(f"  Confidence:   {model.confidence}")
        print(f"  Entry Price:  {model.entry_price}")
        print(f"  Target Price: {model.target_price}")
        print(f"  Stop Loss:    {model.stop_loss}")
        print(f"  Reasoning:    {model.reasoning}")
        print(f"  Valuation:    {model.valuation_assessment}")
        print(f"  Catalysts:    {model.key_catalysts}")
        print(f"  Risks:        {model.key_risks}")
    else:
        print("\nNo structured output available (agent did not produce structured data).")


if __name__ == "__main__":
    asyncio.run(main())
