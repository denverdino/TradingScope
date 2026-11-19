"""Aggressive Risk Debater Agent for TradingScope using AgentScope ReAct framework."""

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


def create_aggressive_debater_agent(
    context: AgentContext,
    name: str = "AggressiveRiskDebater",
) -> Agent:
    """Create an AgentScope ReAct agent for the aggressive risk debater.

    Args:
        model: The language model to use for the agent
        context: AgentContext containing all necessary context information
        name: The name of the agent (default: "AggressiveRiskDebater")

    Returns:
        A configured ReActAgent instance
    """
    ticker = context.ticker

    role_instructions = f"""{SHORT_HORIZON_DECISION_POLICY}

{MARKET_REGIME_DECISION_POLICY}

作为激进风险分析师，评估 Trader 计划中有证据支持的收益机会是否足以补偿风险，不以增长或避免踏空的口号代替下一实际交易日证据。
聚焦保守和中性观点中最关键的分歧，最多2个关键论点，注明来源与数据时点，只引用所需数字并解释建议的新增调整。明确最强反证和撤回或调整立场的条件；对方证据成立时可以承认并修正判断。
只回应已收到的观点，后续轮次只补充新增证据、观点修订与核心分歧；不虚构数据、不重写共享报告。用中文简洁对话，不要求特殊格式或己方获胜。

股票代码：{ticker}"""
    prompt = build_cacheable_system_prompt(
        shared_context=context.generate_risk_evaluation_context_md(),
        role_instructions=role_instructions,
    )

    toolkit = Toolkit()

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
