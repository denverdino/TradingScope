#!/usr/bin/env python3
"""Example showing how to use the RiskManagerAgent with long-term memory."""

import asyncio
import os

from agentscope.model import OpenAIChatModel

from tradingscope.agents.managers.risk_manager import create_risk_manager_agent
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.memory_manager import FinancialMemoryManager


async def main():
    """Example of using the RiskManagerAgent with long-term memory."""
    # Initialize the model
    model = OpenAIChatModel(
        model_name="qwen-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
    )

    # Create AgentContext with sample data
    context = AgentContext()
    context.company_of_interest = "AAPL"

    # Create memory manager for long-term memory
    memory_manager = FinancialMemoryManager()

    try:
        # Create the risk manager agent with long-term memory
        risk_manager = create_risk_manager_agent(
            model=model,
            context=context,
            name="RiskManager",
            long_term_memory=memory_manager.risk_manager_memory,
            long_term_memory_mode="static_control",
        )

        print(f"RiskManagerAgent created: {risk_manager.name}")

        # Example usage with sample data
        sample_prompt = """作为风险管理委员会主席和辩论主持人，您的目标是评估三位风险分析师——激进、中性和安全/保守——之间的辩论，并确定交易员的最佳行动方案。您的决策必须产生明确的建议：买入、卖出或持有。只有在有具体论据强烈支持时才选择持有，而不是在所有方面都似乎有效时作为后备选择。力求清晰和果断。

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

请用中文撰写所有分析内容和建议。"""

        # Call the agent
        response = await risk_manager(sample_prompt)
        print(f"Response: {response.content}")
    finally:
        await memory_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
