"""Risk Manager Agent for TradingScope using AgentScope ReAct framework."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from agentscope import logger

# AgentScope imports
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

# Local imports
from tradingscope.agents.utils.agent_utils import COMPLIANCE_PROMPT
from tradingscope.agents.utils.context import AgentContext

if TYPE_CHECKING:
    from tradingscope.agents.utils.memory import ModelStudioLongTermMemory


def create_risk_manager_agent(
    model: OpenAIChatModel,
    context: AgentContext,
    name: str = "RiskManager",
    long_term_memory: Optional["ModelStudioLongTermMemory"] = None,
    long_term_memory_mode: str = "static_control",
) -> ReActAgent:
    """Create Risk Manager Agent that evaluates risk analysis debates and makes final decisions.

    Args:
        model: AgentScope model instance
        context: AgentContext instance containing all necessary context information
        name: Agent name
        long_term_memory: 长期记忆实例用于存储和检索历史经验
        long_term_memory_mode: 长期记忆模式 ("static_control", "agent_control", "both")

    Returns:
        ReActAgent: Configured risk manager agent
    """
    company_of_interest = context.company_of_interest
    trade_date = context.trade_date

    system_message = f"""{COMPLIANCE_PROMPT}

作为风险管理委员会主席和辩论主持人，您的目标是评估三位风险分析师——激进、中性和安全/保守——之间的辩论，制定最佳交易行动方案。您的决策必须产生明确的建议：买入、卖出或持有。力求决策果断和逻辑清晰。

决策指导原则：
1. **总结关键论点**：提取每位分析师的最强观点，重点关注与背景的相关性。
2. **提供理由**：用辩论中的直接引用和反驳论点支持您的建议。
3. **完善交易计划**：根据分析师的见解调整交易员的原始交易员操作计划。
4. **从过去的错误中学习**：使用长期记忆中的历史经验教训来改进决策，确保不会重复过去的误判。

交付成果：
- 明确且可操作的建议：买入、卖出或持有。
- 基于辩论和过去反思的详细推理。

专注于可操作的见解和持续改进。建立在过去经验教训的基础上，批判性地评估所有观点，确保每个决策都能带来更好的结果。请用中文撰写所有分析内容和建议。

股票代码：{company_of_interest}
交易日期：{trade_date}

# 可用资源：

{context.generate_risk_evaluation_context_md()}"""

    formatter = OpenAIMultiAgentFormatter()
    toolkit = Toolkit()

    # 创建ReActAgent with long-term memory
    agent = ReActAgent(
        name=name,
        sys_prompt=system_message,
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        long_term_memory=long_term_memory,
        long_term_memory_mode=long_term_memory_mode,
        toolkit=toolkit,
        parallel_tool_calls=False,
        max_iters=8,
    )

    logger.debug("📊 [DEBUG] ===== 风险决策经理 Agent 创建完成 =====")
    return agent
