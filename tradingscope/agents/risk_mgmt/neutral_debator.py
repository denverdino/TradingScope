"""Neutral Risk Debator Agent for TradingScope using AgentScope ReAct framework."""

from __future__ import annotations

# AgentScope imports
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

# Local imports
from tradingscope.utils.logging_init import get_logger

logger = get_logger("default")


def create_neutral_debator_agent(
    model: OpenAIChatModel,
    market_research_report: str = "",
    sentiment_report: str = "",
    news_report: str = "",
    fundamentals_report: str = "",
    trader_plan: str = "",
    name: str = "NeutralRiskDebator",
) -> ReActAgent:
    """Create an AgentScope ReAct agent for the neutral risk debator.

    Args:
        model: The language model to use for the agent
        name: The name of the agent (default: "NeutralRiskDebator")

    Returns:
        A configured ReActAgent instance
    """
    formatter = OpenAIMultiAgentFormatter()
    toolkit = Toolkit()

    prompt = f"""作为中性风险分析师，您的角色是提供平衡的视角，权衡交易员决策或计划的潜在收益和风险。您优先考虑全面的方法，评估上行和下行风险，同时考虑更广泛的市场趋势、潜在的经济变化和多元化策略。以下是交易员的决策：

{trader_plan}

您的任务是挑战激进和安全分析师，指出每种观点可能过于乐观或过于谨慎的地方。使用以下数据来源的见解来支持调整交易员决策的温和、可持续策略：

市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新世界事务报告：{news_report}
公司基本面报告：{fundamentals_report}

请不要虚构，只需提出您的观点。

通过批判性地分析双方来积极参与，解决激进和保守论点中的弱点，倡导更平衡的方法。挑战他们的每个观点，说明为什么适度风险策略可能提供两全其美的效果，既提供增长潜力又防范极端波动。专注于辩论而不是简单地呈现数据，旨在表明平衡的观点可以带来最可靠的结果。请用中文以对话方式输出，就像您在说话一样，不使用任何特殊格式。"""

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
