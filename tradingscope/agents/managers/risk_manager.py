"""Risk Manager Agent for TradingScope using AgentScope ReAct framework."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from agentscope import logger

# AgentScope imports
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

# Local imports
from tradingscope.agents.utils.context import AgentContext


def create_risk_manager_agent(
    model: OpenAIChatModel,
    context: AgentContext,
    name: str = "RiskManager",
) -> ReActAgent:
    """Create Risk Manager Agent that evaluates risk analysis debates and makes final decisions."""
    # Extract values from context
    company_of_interest = context.company_of_interest
    market_research_report = context.market_report
    sentiment_report = context.sentiment_report
    news_report = context.news_report
    fundamentals_report = context.fundamentals_report
    trader_plan = context.trader_investment_plan

    system_message = f"""作为风险管理委员会主席和辩论主持人，您的目标是评估三位风险分析师——激进、中性和安全/保守——之间的辩论，并确定交易员的最佳行动方案。您的决策必须产生明确的建议：买入、卖出或持有。只有在有具体论据强烈支持时才选择持有，而不是在所有方面都似乎有效时作为后备选择。力求清晰和果断。

决策指导原则：
1. **总结关键论点**：提取每位分析师的最强观点，重点关注与背景的相关性。
2. **提供理由**：用辩论中的直接引用和反驳论点支持您的建议。
3. **完善交易员计划**：根据分析师的见解调整交易员的原始计划。
4. **从过去的错误中学习**：使用历史经验教训来改进决策，确保不会重复过去的误判。

交付成果：
- 明确且可操作的建议：买入、卖出或持有。
- 基于辩论和过去反思的详细推理。

专注于可操作的见解和持续改进。建立在过去经验教训的基础上，批判性地评估所有观点，确保每个决策都能带来更好的结果。请用中文撰写所有分析内容和建议。


## 可用资源：

公司名称：{company_of_interest}
市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新世界事务新闻：{news_report}
公司基本面报告：{fundamentals_report}
交易员计划：{trader_plan}

"""

    formatter = OpenAIMultiAgentFormatter()
    toolkit = Toolkit()

    agent = ReActAgent(
        name=name,
        sys_prompt=system_message,
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        toolkit=toolkit,
        parallel_tool_calls=False,
        max_iters=8,
    )

    logger.debug("📊 [DEBUG] ===== 风险决策经理 Agent 创建完成 =====")
    return agent
