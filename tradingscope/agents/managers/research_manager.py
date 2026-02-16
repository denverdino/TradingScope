from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from agentscope import logger
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field

# Import unified logging system
from tradingscope.agents.utils.agent_utils import COMPLIANCE_PROMPT
from tradingscope.agents.utils.context import AgentContext

if TYPE_CHECKING:
    from tradingscope.agents.utils.memory import ModelStudioLongTermMemory


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
    model: OpenAIChatModel,
    context: AgentContext,
    name: str = "ResearchManager",
    long_term_memory: Optional["ModelStudioLongTermMemory"] = None,
    long_term_memory_mode: str = "static_control",
) -> ReActAgent:
    """创建支持结构化输出的投资决策经理 ReActAgent。

    Args:
        model: AgentScope 模型实例
        context: AgentContext实例包含所有必要的上下文信息
        name: 代理名称
        long_term_memory: 长期记忆实例用于存储和检索历史经验
        long_term_memory_mode: 长期记忆模式 ("static_control", "agent_control", "both")

    Returns:
        ReActAgent: 配置好的投资决策经理代理
    """
    company_of_interest = context.company_of_interest
    trade_date = context.trade_date

    # 构建系统提示词，鼓励结构化输出
    system_message = f"""{COMPLIANCE_PROMPT}

作为投资组合经理和辩论主持人，您的职责是批判性地评估这轮辩论并做出明确决策：支持看跌分析师、看涨分析师，或者仅在基于所提出论点有强有力理由时选择持有。
简洁地总结双方的关键观点，重点关注最有说服力的证据或推理。您的建议——买入、卖出或持有——必须明确且可操作。
避免仅仅因为双方都有有效观点就默认选择持有；要基于辩论中最强有力的论点做出承诺。

此外，为交易员制定详细的投资计划。这应该包括
建议：基于最有说服力论点的明确立场。
理由：解释为什么这些论点导致您的结论。
战略行动：实施建议的具体步骤。
📊 目标价格分析：基于所有可用报告（基本面、新闻、情绪），提供全面的目标价格区间和具体价格目标。考虑：
    - 基本面报告中的基本估值
    - 新闻对价格预期的影响
    - 情绪驱动的价格调整
    - 技术支撑/阻力位
    - 价格目标的时间范围
        - 1天: 价格情景（保守、基准、乐观）
        - 5天: 价格情景（保守、基准、乐观）
        - 1个月: 价格情景（保守、基准、乐观）

**注意：**

- 您必须提供具体的目标价格 - 不要回复'无法确定'或'需要更多信息'。
- 考虑您在类似情况下的过去错误。利用长期记忆中的历史经验教训来完善您的决策制定，确保您在学习和改进。
- 请用中文撰写所有分析内容和建议。

**重要：您的回复必须以如下格式开头：**

```
股票代码：{company_of_interest}
交易日期：{trade_date}
交易决策：【买入/卖出/持有】
```

然后再提供详细分析。

# 可用资源：

{context.generate_analyst_reports_md()}"""

    # 工具注册（如果需要）
    formatter = OpenAIMultiAgentFormatter()
    toolkit = Toolkit()

    # 创建 ReActAgent with long-term memory
    agent = ReActAgent(
        name=name,
        sys_prompt=system_message,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        long_term_memory=long_term_memory,
        long_term_memory_mode=long_term_memory_mode,
        parallel_tool_calls=True,
        max_iters=8,
    )

    logger.debug("📊 [DEBUG] ===== 结构化投资决策经理 Agent 创建完成 =====")
    return agent
