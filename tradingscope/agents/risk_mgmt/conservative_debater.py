"""Conservative Risk Debater Agent for TradingScope using AgentScope ReAct framework."""

from __future__ import annotations

from agentscope import logger

# AgentScope imports
from agentscope.agent import Agent
from agentscope.agent._config import ReActConfig
from agentscope.tool import Toolkit

# Local imports
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.decision_policy import MARKET_REGIME_DECISION_POLICY, SHORT_HORIZON_DECISION_POLICY
from tradingscope.agents.utils.prompt_cache import build_cacheable_system_prompt


def create_conservative_debater_agent(
    context: AgentContext,
    name: str = "ConservativeRiskDebater",
) -> Agent:
    """Create an AgentScope agent for the conservative risk debater.

    Args:
        model: The language model to use for the agent
        context: AgentContext containing all necessary context information
        name: The name of the agent (default: "ConservativeRiskDebater")

    Returns:
        A configured agent instance
    """
    ticker = context.ticker
    toolkit = Toolkit()

    role_instructions = f"""{SHORT_HORIZON_DECISION_POLICY}

{MARKET_REGIME_DECISION_POLICY}

作为安全/保守风险分析师，您的主要目标是基于下一实际交易日直至收盘的证据评估资产保护、波动性和下行风险。您优先考虑稳定性、安全性和风险缓解，但不能把长期增长或保护资本的口号作为本次方向证据。在评估交易员的交易操作计划时，请批判性地审查高风险要素，说明更谨慎的替代方案如何应对目标交易日的损失与隔夜跳空风险；独立的持仓管理期限、止损和仓位安排可以延续到收盘之后，不得因此延长预测窗口或要求收盘强制平仓。
聚焦激进和中性观点中最关键的分歧，最多2个关键论点，注明来源与数据时点，只引用所需数字并解释风险控制的新增调整。明确最强反证和撤回或调整立场的条件；对方证据成立时可以承认并修正判断。
只回应已收到的观点，后续轮次只补充新增证据、观点修订与核心分歧；不虚构数据、不重写共享报告。用中文简洁对话，不要求特殊格式或己方获胜。

股票代码：{ticker}"""
    prompt = build_cacheable_system_prompt(
        shared_context=context.generate_risk_evaluation_context_md(),
        role_instructions=role_instructions,
    )

    # Create the agent
    agent = Agent(
        name=name,
        system_prompt=prompt,
        model=context.non_thinking_model,
        middlewares=context.middlewares,
        toolkit=toolkit,
        react_config=ReActConfig(max_iters=5),
    )

    return agent
