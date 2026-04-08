#!/usr/bin/env python3
"""Example script demonstrating how to use the Research Manager Agent with long-term memory."""

import asyncio

from tradingscope.agents.managers.research_manager import create_research_manager_agent
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.memory_manager import FinancialMemoryManager


async def main():
    # Create AgentContext with sample data
    context = AgentContext()
    context.company_of_interest = "AAPL"

    # Create memory manager for long-term memory
    memory_manager = FinancialMemoryManager()

    try:
        # Create the research manager agent with long-term memory
        agent = create_research_manager_agent(
            context=context,
            long_term_memory=memory_manager.research_manager_memory,
            long_term_memory_mode="static_control",
        )

        # Example usage with sample data
        sample_prompt = """作为投资组合经理和辩论主持人，您的职责是批判性地评估这轮辩论并做出明确决策：支持看跌分析师、看涨分析师，或者仅在基于所提出论点有强有力理由时选择持有。

请基于以下信息做出决策：

市场研究：科技股整体呈现上涨趋势，但波动性增加。人工智能和云计算板块表现突出。

情绪分析：社交媒体上对科技股的情绪偏向积极，但有部分投资者表达对估值过高的担忧。

新闻分析：最新财报显示几家大型科技公司业绩超预期，但监管政策可能对行业发展产生影响。

基本面分析：目标公司市盈率较高，但营收增长稳定，研发投入占比大。

辩论历史：
看涨分析师：基于强劲的业绩和行业发展趋势，建议买入。
看跌分析师：考虑到高估值和潜在的监管风险，建议卖出。

请用中文撰写所有分析内容和建议。"""

        # Run the agent with the sample prompt
        response = await agent(sample_prompt)
        print("Investment Decision:", response.content)
    finally:
        await memory_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
