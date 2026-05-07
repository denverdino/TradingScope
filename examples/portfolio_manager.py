#!/usr/bin/env python3
"""Example showing how to use the PortfolioManagerAgent with structured output."""

import asyncio

from agentscope.message import Msg

from tradingscope.agents.managers.portfolio_manager import create_portfolio_manager_agent
from tradingscope.agents.output import PortfolioStructuredOutput
from tradingscope.agents.utils.context import AgentContext


async def main():
    """Example of using the PortfolioManagerAgent with structured output."""
    # Create AgentContext with sample data
    context = AgentContext()
    context.company_of_interest = "AAPL"

    # Create the portfolio manager agent
    portfolio_manager = create_portfolio_manager_agent(
        context=context,
        name="PortfolioManager",
    )

    print(f"PortfolioManagerAgent created: {portfolio_manager.name}")

    # Example usage with sample data (prompt must be a Msg object)
    prompt = Msg(
        name="user",
        role="user",
        content="""作为风险管理委员会主席和辩论主持人，您的目标是评估三位风险分析师——激进、中性和安全/保守——之间的辩论，并确定交易员的最佳行动方案。您的决策必须产生明确的建议：买入、卖出或持有。只有在有具体论据强烈支持时才选择持有，而不是在所有方面都似乎有效时作为后备选择。力求清晰和果断。

请基于以下信息做出决策：

市场研究：科技股整体呈现上涨趋势，但波动性增加。人工智能和云计算板块表现突出。

情绪分析：社交媒体上对科技股的情绪偏向积极，但有部分投资者表达对估值过高的担忧。

新闻分析：最新财报显示几家大型科技公司业绩超预期，但监管政策可能对行业发展产生影响。

基本面分析：目标公司市盈率较高，但营收增长稳定，研发投入占比大。

交易员计划：计划买入AAPL股票100股，目标价位200美元，止损价位180美元。

辩论历史：
激进分析师：基于强劲的业绩和行业发展趋势，建议买入。
保守分析师：考虑到高估值和潜在的监管风险，建议卖出。
中性分析师：建议持有观察，等待更明确的市场信号。

请用中文撰写所有分析内容和建议。""",
    )

    # Call the agent with structured output
    response = await portfolio_manager(prompt, structured_model=PortfolioStructuredOutput)

    print("=" * 60)
    print("Portfolio Manager Response:")
    print(response.content)
    print("=" * 60)

    structured = response.metadata if response.metadata and ("direction" in response.metadata or "action" in response.metadata) else None
    if structured:
        print("\nStructured Output:")
        model = PortfolioStructuredOutput.model_validate(structured)
        print(f"  Direction:             {model.direction}")
        print(f"  Action:                {model.action}")
        print(f"  Confidence:            {model.confidence}")
        print(f"  Entry Price:           {model.entry_price}")
        print(f"  Target Price:          {model.target_price}")
        print(f"  Stop Loss:             {model.stop_loss}")
        print(f"  Position Advice:       {model.position_advice}")
        print(f"  Risk Score:            {model.risk_score}")
        print(f"  Aggressive Viewpoint:  {model.aggressive_viewpoint}")
        print(f"  Conservative Viewpoint:{model.conservative_viewpoint}")
        print(f"  Neutral Viewpoint:     {model.neutral_viewpoint}")
        print(f"  Adopted Reasoning:     {model.adopted_reasoning}")
        print(f"  Risk Control Measures: {model.risk_control_measures}")
        print(f"  Invalidation:          {model.invalidation_conditions}")
        print(f"  Reasoning:             {model.reasoning}")
    else:
        print("\nNo structured output available (agent did not produce structured data).")


if __name__ == "__main__":
    asyncio.run(main())
