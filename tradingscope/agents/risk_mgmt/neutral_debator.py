"""Neutral Risk Debator Agent for TradingScope using AgentScope ReAct framework."""

from __future__ import annotations

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


def create_neutral_debator_agent(
    model: OpenAIChatModel,
    context: AgentContext,
    name: str = "NeutralRiskDebator",
) -> ReActAgent:
    """Create an AgentScope ReAct agent for the neutral risk debator.

    Args:
        model: The language model to use for the agent
        context: AgentContext containing all necessary context information
        name: The name of the agent (default: "NeutralRiskDebator")

    Returns:
        A configured ReActAgent instance
    """
    company_of_interest = context.company_of_interest
    formatter = OpenAIMultiAgentFormatter()
    toolkit = Toolkit()

    prompt = f"""{COMPLIANCE_PROMPT}

作为中性风险分析师，您的角色是提供平衡的视角，权衡交易员交易操作计划的潜在收益和风险。您优先考虑全面的方法，评估上行和下行风险，同时考虑更广泛的市场趋势、潜在的经济变化和多元化策略。
您的任务是挑战激进和安全分析师，指出每种观点可能过于乐观或过于谨慎的地方。使用以下数据来源的见解来支持调整交易员交易操作计划的温和、可持续策略：
请不要虚构，只需提出您的观点。

通过批判性地分析双方来积极参与，解决激进和保守论点中的弱点，倡导更平衡的方法。挑战他们的每个观点，说明为什么适度风险策略可能提供两全其美的效果，既提供增长潜力又防范极端波动。专注于辩论而不是简单地呈现数据，旨在表明平衡的观点可以带来最可靠的结果。请用中文以对话方式输出，就像您在说话一样，不使用任何特殊格式。

公司名称：{company_of_interest}

# 可用资源：

{context.generate_risk_evaluation_context_md()}"""


    # Create the agent
    agent = ReActAgent(
        name=name,
        sys_prompt=prompt,
        model=model,
        formatter=formatter,
        memory=InMemoryMemory(),
        toolkit=toolkit,
        parallel_tool_calls=False,
        max_iters=5,
    )

    return agent
