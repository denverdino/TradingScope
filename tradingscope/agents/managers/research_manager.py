from __future__ import annotations

from enum import Enum

from agentscope import logger
from agentscope.agent import Agent
from agentscope.agent._config import ReActConfig
from pydantic import BaseModel, Field

# Import unified logging system
from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.utils.decision_policy import MARKET_REGIME_DECISION_POLICY, SHORT_HORIZON_DECISION_POLICY
from tradingscope.agents.utils.prompt_cache import build_cacheable_system_prompt


class RecommendationEnum(str, Enum):
    BUY = "买入"
    SELL = "卖出"
    HOLD = "持有"


class PriceScenario(BaseModel):
    scenario: str = Field(description="情景名称，如'保守', '基准', '乐观'")
    target_price: float = Field(description="目标价格")
    timeframe_months: int = Field(description="时间范围(月)")


class InvestmentDecision(BaseModel):
    recommendation: RecommendationEnum = Field(description="投资建议：买入/卖出/持有")
    summary: str = Field(description="双方关键观点的简洁总结")
    reasoning: str = Field(description="解释为什么这些论点导致结论")
    strategic_actions: list[str] = Field(description="实施建议的具体步骤")
    price_analysis: str = Field(description="目标价格分析")
    price_scenarios: list[PriceScenario] = Field(description="不同情景下的价格目标")
    target_price_range: tuple[float, float] = Field(description="目标价格区间")


def create_research_manager_agent(
    context: AgentContext,
    name: str = "ResearchManager",
) -> Agent:
    """创建支持结构化输出的投资决策经理 Agent。

    Args:
        context: AgentContext实例包含所有必要的上下文信息
        name: 代理名称

    Returns:
        Agent: 配置好的投资决策经理代理
    """
    company_of_interest = context.company_of_interest
    trade_date = context.trade_date
    latest_trading_date = context.latest_trading_date

    # 构建系统提示词，鼓励结构化输出
    role_instructions = f"""{SHORT_HORIZON_DECISION_POLICY}

{MARKET_REGIME_DECISION_POLICY}

作为头部科技股研究经理和辩论主持人，您的职责是批判性地评估这轮辩论并做出明确的短周期研究判断：支持看跌分析师、看涨分析师，或者仅在基于所提出论点有强有力理由时选择持有。
简洁地总结双方的关键观点，重点关注最有说服力的证据或推理。您的建议——买入、卖出或持有——必须明确且可操作。
避免仅仅因为双方都有有效观点就默认选择持有；要基于辩论中最强有力的论点做出承诺。

此外，为交易员提供情景分析，不制定完整订单。这应该包括
建议：基于最有说服力论点的明确立场。
理由：解释为什么这些论点导致您的结论，并注明关键证据来源、事件去重结果和最强反证。
战略行动：给出需要观察的触发条件和失效条件，不要替 Trader 虚构完整入场、目标、止损或时间止损订单。
📊 目标价格分析：基于所有可用报告（基本面、新闻、情绪），提供全面的目标价格区间和具体价格目标。考虑：
    - 基本面报告中的基本估值
    - 新闻对价格预期的影响
    - 情绪驱动的价格调整
    - 技术支撑/阻力位
    - 价格目标的时间范围
        - 1天: 价格情景（保守、基准、乐观）
        - 5天: 价格情景（保守、基准、乐观）

**注意：**

- 您必须提供具体的目标价格 - 不要回复'无法确定'或'需要更多信息'。
- 考虑您在类似情况下的过去错误。利用长期记忆中的历史经验教训来完善您的决策制定，确保您在学习和改进。
- 请用中文撰写所有分析内容和建议。

**重要：您的回复必须以如下格式开头：**

- **股票代码**：{company_of_interest}
- **交易日期**：{trade_date}
- **最新美股交易日期**：{latest_trading_date}
- **交易决策**：【买入/卖出/持有】


然后按以下格式提供详细分析：

### 决策摘要
- 置信度：0.x（0-1之间）
- 核心理由：1-2句话总结

### 多空观点总结
- **看涨论点**：列出2-3个关键看涨观点
- **看跌论点**：列出2-3个关键看跌观点
- **决策依据**：解释为何采纳某方观点

### 目标价格分析
- **短期目标（1-5天）**：保守/基准/乐观价格
- **价格区间**：xxx - xxx
- **关键支撑/阻力位**：列出关键价位

### 战略行动建议
- 列出3-5条观察条件、情景触发器和失效条件；不制定完整订单

### 风险提示
- 列出2-3个主要风险点

请输出简洁但完整的研究素材，覆盖事实依据、来源去重、最强反证、关键指标、风险、情景结论与失效条件。
不要输出 JSON、JSON 代码块或固定 Markdown 报告模板；系统将在下一阶段根据严格 schema 生成正式结果。"""
    system_message = build_cacheable_system_prompt(
        shared_context=context.generate_analyst_reports_md(),
        role_instructions=role_instructions,
    )

    # 创建 Agent
    agent = Agent(
        name=name,
        system_prompt=system_message,
        model=context.model,
        middlewares=context.middlewares,
        react_config=ReActConfig(max_iters=8),
    )

    logger.debug("📊 [DEBUG] ===== 结构化投资决策经理 Agent 创建完成 =====")
    return agent
