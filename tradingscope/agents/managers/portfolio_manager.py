"""Portfolio Manager Agent for TradingScope using AgentScope framework."""

from __future__ import annotations

from agentscope import logger

# AgentScope imports
from agentscope.agent import Agent

# Local imports
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.decision_policy import (
    EXECUTION_PLAN_DECISION_POLICY,
    MARKET_REGIME_DECISION_POLICY,
    SHORT_HORIZON_DECISION_POLICY,
)
from tradingscope.agents.utils.prompt_cache import build_cacheable_system_prompt


def create_portfolio_manager_agent(
    context: AgentContext,
    name: str = "PortfolioManager",
) -> Agent:
    """Create Portfolio Manager Agent that evaluates risk analysis debates and makes final decisions.

    Args:
        context: AgentContext instance containing all necessary context information
        name: Agent name

    Returns:
        Agent: Configured portfolio manager agent
    """
    ticker = context.ticker
    analysis_date = context.analysis_date
    last_completed_session = context.last_completed_session

    role_instructions = f"""{SHORT_HORIZON_DECISION_POLICY}

{MARKET_REGIME_DECISION_POLICY}

{EXECUTION_PLAN_DECISION_POLICY}

作为头部科技股风险管理委员会主席和辩论主持人，您的目标是评估三位风险分析师——激进、中性和安全/保守——之间的辩论，制定最佳交易行动方案。

用中文给出最终方向、买入、卖出或持有操作、置信度与一句话摘要；基于三位辩手的关键分歧说明证据取舍，不逐人重写论证。
完善 Trader 计划时，只解释采纳、拒绝或新增调整及其可核验来源、事件去重、最强反证和失效条件；事件驱动状态切换不强制等待右侧价格确认。最终计划必须完整保留适用价格与执行步骤，而非仅引用上游订单。

分析素材应覆盖以下内容，无需遵循固定展示格式：
- 股票代码：{ticker}；分析日期：{analysis_date}；最近完成交易日：{last_completed_session}。
- 唯一方向与操作建议、置信度（0-1）、风险评分（0-1）及核心理由。
- 三位辩手的关键观点与采纳/拒绝理由，关联证据来源、原始链接（如有）、数据日期、最强反证和失效条件。
- 所有价格与金额注明货币；价格无依据时留空（null）并说明限制。
- 数据缺失与适用限制。

可执行价格计划素材：
- 先说明交易前持仓假设和执行后的 `position_advice`，显式给出唯一 `trade_intent`，按 trade_intent 只填写适用字段，遵循上方六类交易意图与可执行价格规则，不再重列意图清单。
- 给出最终适用的价格与区间、依据、执行步骤、减仓比例和持仓管理期限；明确相对 Trader 计划的保留或调整，不因精简省略价格计划。
- 风险控制措施及失效条件按实际需要完整保留，不受关键证据条数限制。

请输出简洁但完整的分析素材，覆盖事实依据、关键指标、风险、结论与可执行价格计划；核心理由最多3点，不得重抄共享报告，只引用决策所需的关键数据。
不要输出 JSON、JSON 代码块或固定 Markdown 报告模板；系统将在下一阶段根据严格 schema 生成正式结果。"""
    system_message = build_cacheable_system_prompt(
        shared_context=context.generate_risk_evaluation_context_md(),
        role_instructions=role_instructions,
    )

    agent = Agent(
        name=name,
        system_prompt=system_message,
        model=context.model,
        middlewares=context.middlewares,
    )

    logger.debug("📊 [DEBUG] ===== 投资组合经理 Agent 创建完成 =====")
    return agent
