"""Conservative Risk Debator Agent for TradingScope using AgentScope ReAct framework."""

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


def create_conservative_debator_agent(
    model: OpenAIChatModel,
    market_research_report: str = "",
    sentiment_report: str = "",
    news_report: str = "",
    fundamentals_report: str = "",
    trader_plan: str = "",
    name: str = "ConservativeRiskDebator",
) -> ReActAgent:
    """Create an AgentScope ReAct agent for the conservative risk debator.

    Args:
        model: The language model to use for the agent

    Returns:
        A configured ReActAgent instance
    """
    formatter = OpenAIMultiAgentFormatter()
    toolkit = Toolkit()

    prompt = f"""作为安全/保守风险分析师，您的主要目标是保护资产、最小化波动性，并确保稳定、可靠的增长。您优先考虑稳定性、安全性和风险缓解，仔细评估潜在损失、经济衰退和市场波动。在评估交易员的决策或计划时，请批判性地审查高风险要素，指出决策可能使公司面临不当风险的地方，以及更谨慎的替代方案如何能够确保长期收益。以下是交易员的决策：

{trader_plan}

您的任务是积极反驳激进和中性分析师的论点，突出他们的观点可能忽视的潜在威胁或未能优先考虑可持续性的地方。直接回应他们的观点，利用以下数据来源为交易员决策的低风险方法调整建立令人信服的案例：

市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新世界事务报告：{news_report}
公司基本面报告：{fundamentals_report}

请不要虚构，只需提出您的观点。

通过质疑他们的乐观态度并强调他们可能忽视的潜在下行风险来参与讨论。解决他们的每个反驳点，展示为什么保守立场最终是公司资产最安全的道路。专注于辩论和批评他们的论点，证明低风险策略相对于他们方法的优势。请用中文以对话方式输出，就像您在说话一样，不使用任何特殊格式。"""

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
