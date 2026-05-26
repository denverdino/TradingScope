"""Conservative Risk Debator Agent for TradingScope using AgentScope ReAct framework."""

from __future__ import annotations

from agentscope import logger

# AgentScope imports
from agentscope.agent import Agent
from agentscope.agent._config import ReActConfig
from agentscope.tool import Toolkit

# Local imports
from tradingscope.agents.utils.agent_utils import COMPLIANCE_PROMPT
from tradingscope.agents.utils.context import AgentContext


def create_conservative_debator_agent(
    context: AgentContext,
    name: str = "ConservativeRiskDebator",
) -> Agent:
    """Create an AgentScope agent for the conservative risk debator.

    Args:
        model: The language model to use for the agent
        context: AgentContext containing all necessary context information
        name: The name of the agent (default: "ConservativeRiskDebator")

    Returns:
        A configured agent instance
    """
    company_of_interest = context.company_of_interest
    toolkit = Toolkit()

    prompt = f"""{COMPLIANCE_PROMPT}

作为安全/保守风险分析师，您的主要目标是保护资产、最小化波动性，并确保稳定、可靠的增长。您优先考虑稳定性、安全性和风险缓解，仔细评估潜在损失、经济衰退和市场波动。在评估交易员的交易操作计划时，请批判性地审查高风险要素，指出计划中可能使公司面临不当风险的地方，以及更谨慎的替代方案如何能够确保长期收益。
您的任务是积极反驳激进和中性分析师的论点，突出他们的观点可能忽视的潜在威胁或未能优先考虑可持续性的地方。直接回应他们的观点，利用以下数据来源为交易员操作计划的低风险方法调整建立令人信服的案例：
请不要虚构，只需提出您的观点。
通过质疑他们的乐观态度并强调他们可能忽视的潜在下行风险来参与讨论。解决他们的每个反驳点，展示为什么保守立场最终是公司资产最安全的道路。专注于辩论和批评他们的论点，证明低风险策略相对于他们方法的优势。请用中文以对话方式输出，就像您在说话一样，不使用任何特殊格式。

公司名称：{company_of_interest}

# 可用资源：

{context.generate_risk_evaluation_context_md()}"""

    # Create the agent
    agent = Agent(
        name=name,
        system_prompt=prompt,
        model=context.non_thinking_model,
        toolkit=toolkit,
        react_config=ReActConfig(max_iters=5),
    )

    return agent
