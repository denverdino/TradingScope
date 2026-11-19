"""Neutral Risk Debater Agent for TradingScope using AgentScope ReAct framework."""

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


def create_neutral_debater_agent(
    context: AgentContext,
    name: str = "NeutralRiskDebater",
) -> Agent:
    """Create an AgentScope agent for the neutral risk debater.

    Args:
        model: The language model to use for the agent
        context: AgentContext containing all necessary context information
        name: The name of the agent (default: "NeutralRiskDebater")

    Returns:
        A configured agent instance
    """
    ticker = context.ticker
    toolkit = Toolkit()

    role_instructions = f"""{SHORT_HORIZON_DECISION_POLICY}

{MARKET_REGIME_DECISION_POLICY}

作为中性风险分析师，基于下一实际交易日证据权衡 Trader 计划的收益与风险，不为折中而默认持有。
聚焦激进和保守观点中最关键的分歧，最多2个关键论点，注明来源与数据时点，只引用所需数字并解释建议的新增调整。明确最强反证和撤回或调整立场的条件；对方证据成立时可以承认并修正判断。
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
