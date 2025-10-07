from __future__ import annotations

from enum import Enum

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field

# Import unified logging system
from tradingscope.utils.logging_init import get_logger

logger = get_logger("default")


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
    name: str = "ResearchManager",
) -> ReActAgent:
    """创建支持结构化输出的投资决策经理 ReActAgent。

    Args:
        model: AgentScope 模型实例
        memory: 记忆模块实例（可选）

    Returns:
        ReActAgent: 配置好的投资决策经理代理
    """

    # 构建系统提示词，鼓励结构化输出
    system_message = (
        "作为投资组合经理和辩论主持人，您的职责是批判性地评估这轮辩论并做出明确决策：支持看跌分析师、看涨分析师，"
        "或者仅在基于所提出论点有强有力理由时选择持有。\\n\\n"
        "简洁地总结双方的关键观点，重点关注最有说服力的证据或推理。您的建议——买入、卖出或持有——必须明确且可操作。"
        "避免仅仅因为双方都有有效观点就默认选择持有；要基于辩论中最强有力的论点做出承诺。\\n\\n"
        "此外，为交易员制定详细的投资计划。这应该包括：\\n\\n"
        "您的建议：基于最有说服力论点的明确立场。\\n"
        "理由：解释为什么这些论点导致您的结论。\\n"
        "战略行动：实施建议的具体步骤。\\n"
        "📊 目标价格分析：基于所有可用报告（基本面、新闻、情绪），提供全面的目标价格区间和具体价格目标。考虑：\\n"
        "- 基本面报告中的基本估值\\n"
        "- 新闻对价格预期的影响\\n"
        "- 情绪驱动的价格调整\\n"
        "- 技术支撑/阻力位\\n"
        "- 风险调整价格情景（保守、基准、乐观）\\n"
        "- 价格目标的时间范围（1个月、3个月、6个月）\\n"
        "💰 您必须提供具体的目标价格 - 不要回复'无法确定'或'需要更多信息'。\\n\\n"
        "考虑您在类似情况下的过去错误。利用这些见解来完善您的决策制定，确保您在学习和改进。"
        "以对话方式呈现您的分析，但在最后请尝试以结构化的要点形式总结关键信息，便于后续处理。\\n\\n"
        "请用中文撰写所有分析内容和建议。"
    )

    # 工具注册（如果需要）
    formatter = OpenAIMultiAgentFormatter()
    toolkit = Toolkit()

    # 创建模型与 Agent
    agent = ReActAgent(
        name=name,
        sys_prompt=system_message,
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        parallel_tool_calls=True,
        max_iters=8,
    )

    logger.debug("📊 [DEBUG] ===== 结构化投资决策经理 Agent 创建完成 =====")
    return agent
