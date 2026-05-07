#!/usr/bin/env python3
"""Example script demonstrating how to use the Trader Agent with structured output."""

import asyncio

from tradingscope.agents.output import TraderStructuredOutput
from tradingscope.agents.trader.trader import create_trader_agent
from tradingscope.agents.utils.context import AgentContext


async def main():
    # Create sample data for testing
    company_of_interest = "AAPL"
    investment_plan = "Based on technical analysis, consider entering a long position with a target price of $200."
    market_research_report = "Technical indicators show a bullish trend with strong support at $180."
    sentiment_report = "Social media sentiment is positive with increasing mentions of the company."
    news_report = "Recent news indicates strong quarterly earnings and product launches."
    fundamentals_report = "The company has strong fundamentals with a P/E ratio of 25 and consistent revenue growth."

    # Create AgentContext (trade_date defaults to today)
    context = AgentContext()
    context.company_of_interest = company_of_interest
    context.investment_plan = investment_plan
    context.market_report = market_research_report
    context.sentiment_report = sentiment_report
    context.news_report = news_report
    context.fundamentals_report = fundamentals_report

    # Create the trader agent
    agent = create_trader_agent(
        context=context,
    )

    # Run the agent with structured output
    response = await agent(None, structured_model=TraderStructuredOutput)

    print("=" * 60)
    print("Trader Response:")
    print(response.content)
    print("=" * 60)

    structured = response.metadata if response.metadata and ("direction" in response.metadata or "action" in response.metadata) else None
    if structured:
        print("\nStructured Output:")
        model = TraderStructuredOutput.model_validate(structured)
        print(f"  Direction:          {model.direction}")
        print(f"  Action:             {model.action}")
        print(f"  Confidence:         {model.confidence}")
        print(f"  Entry Price:        {model.entry_price}")
        print(f"  Target Price:       {model.target_price}")
        print(f"  Stop Loss:          {model.stop_loss}")
        print(f"  Risk Reward Ratio:  {model.risk_reward_ratio}")
        print(f"  Position Advice:    {model.position_advice}")
        print(f"  Risk Score:         {model.risk_score}")
        print(f"  Time Stop Days:     {model.time_stop_days}")
        print(f"  Invalidation:       {model.invalidation_conditions}")
        print(f"  Reasoning:          {model.reasoning}")
    else:
        print("\nNo structured output available (agent did not produce structured data).")


if __name__ == "__main__":
    asyncio.run(main())
