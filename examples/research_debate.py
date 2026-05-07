#!/usr/bin/env python3
"""Example script demonstrating the research multi-agent debate system with structured output."""

import asyncio
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tradingscope.agents.managers.research_manager import create_research_manager_agent
from tradingscope.agents.output import ResearchManagerStructuredOutput
from tradingscope.agents.researchers.bear_researcher import create_bear_researcher_agent
from tradingscope.agents.researchers.bull_researcher import create_bull_researcher_agent
from tradingscope.agents.researchers.debate_orchestrator import create_research_debate_orchestrator
from tradingscope.agents.utils.context import AgentContext


async def main():
    """Example usage of the research debate system with structured output."""

    # Create AgentContext
    context = AgentContext()
    context.company_of_interest = "AAPL"
    context.market_report = "Technical indicators show a bullish trend with strong support at $180."
    context.sentiment_report = "Social media sentiment is positive with increasing mentions of the company."
    context.news_report = "Recent news indicates strong quarterly earnings and product launches."
    context.fundamentals_report = "The company has strong fundamentals with a P/E ratio of 25 and consistent revenue growth."

    # Create the research agents
    bull_researcher = create_bull_researcher_agent(
        context=context,
    )

    bear_researcher = create_bear_researcher_agent(
        context=context,
    )

    research_manager = create_research_manager_agent(
        context=context,
    )

    # Create the debate orchestrator with structured output
    orchestrator = create_research_debate_orchestrator(
        bull_researcher=bull_researcher,
        bear_researcher=bear_researcher,
        research_manager=research_manager,
        max_rounds=3,
        research_structured_model=ResearchManagerStructuredOutput,
    )

    print(f"Starting research debate for {context.company_of_interest}")
    print("=" * 60)

    # Run the debate
    final_decision = await orchestrator.run_debate(
        company_name=context.company_of_interest,
    )

    print("=" * 60)
    print("Final Decision from Research Manager:")
    print(final_decision.content)
    print("=" * 60)

    structured = (
        final_decision.metadata if final_decision.metadata and ("direction" in final_decision.metadata or "action" in final_decision.metadata) else None
    )
    if structured:
        print("\nStructured Output:")
        model = ResearchManagerStructuredOutput.model_validate(structured)
        print(f"  Direction:         {model.direction}")
        print(f"  Action:            {model.action}")
        print(f"  Confidence:        {model.confidence}")
        print(f"  Entry Price:       {model.entry_price}")
        print(f"  Target Price:      {model.target_price}")
        print(f"  Stop Loss:         {model.stop_loss}")
        print(f"  Bull Viewpoints:   {model.bull_viewpoints}")
        print(f"  Bear Viewpoints:   {model.bear_viewpoints}")
        print(f"  Adopted Reasoning: {model.adopted_reasoning}")
        print(f"  Reasoning:         {model.reasoning}")
    else:
        print("\nNo structured output available (agent did not produce structured data).")


if __name__ == "__main__":
    asyncio.run(main())
