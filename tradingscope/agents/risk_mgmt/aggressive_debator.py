"""Aggressive Risk Debator Agent for TradingScope using AgentScope ReAct framework."""

from __future__ import annotations

from agentscope import logger

# AgentScope imports
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

# Local imports
from tradingscope.agents.utils.context import AgentContext


def create_aggressive_debator_agent(
    model: OpenAIChatModel,
    context: AgentContext,
    name: str = "AggressiveRiskDebator",
) -> ReActAgent:
    """Create an AgentScope ReAct agent for the aggressive risk debator.

    Args:
        model: The language model to use for the agent
        context: AgentContext containing all necessary context information
        name: The name of the agent (default: "AggressiveRiskDebator")

    Returns:
        A configured ReActAgent instance
    """
    company_of_interest = context.company_of_interest

    prompt = f"""作为激进风险分析师，您的职责是积极倡导高回报、高风险的投资机会，强调大胆策略和竞争优势。在评估交易员的决策或计划时，请重点关注潜在的上涨空间、增长潜力和创新收益——即使这些伴随着较高的风险。使用提供的市场数据和情绪分析来加强您的论点，并挑战对立观点。具体来说，请直接回应保守和中性分析师提出的每个观点，用数据驱动的反驳和有说服力的推理进行反击。突出他们的谨慎态度可能错过的关键机会，或者他们的假设可能过于保守的地方。

您的任务是通过质疑和批评保守和中性立场来为交易员的决策创建一个令人信服的案例，证明为什么您的高回报视角提供了最佳的前进道路。将以下来源的见解纳入您的论点：
请不要虚构，只需提出您的观点。
积极参与，解决提出的任何具体担忧，反驳他们逻辑中的弱点，并断言承担风险的好处以超越市场常规。专注于辩论和说服，而不仅仅是呈现数据。挑战每个反驳点，强调为什么高风险方法是最优的。请用中文以对话方式输出，就像您在说话一样，不使用任何特殊格式。

## 可用资源：
公司名称：{company_of_interest}

{context.generate_all_reports_md()}

## 交易员计划
{context.trader_investment_plan}
"""

    formatter = OpenAIMultiAgentFormatter()
    toolkit = Toolkit()

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
