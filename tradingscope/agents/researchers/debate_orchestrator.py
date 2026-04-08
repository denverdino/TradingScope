"""Multi-Agent Debate Orchestrator for Bull/Bear Researchers using AgentScope."""

from __future__ import annotations

# Local imports
from agentscope import logger

# AgentScope imports
from agentscope.message import Msg
from agentscope.pipeline import MsgHub

from tradingscope.agents.utils.agent_utils import call_agent_with_retry


class ResearchDebateOrchestrator:
    """Orchestrates a multi-agent debate between bull and bear researchers."""

    def __init__(
        self,
        bull_researcher,
        bear_researcher,
        research_manager,
        max_rounds: int = 3,
    ):
        """Initialize the debate orchestrator.

        Args:
            bull_researcher: Bull researcher agent
            bear_researcher: Bear researcher agent
            research_manager: Research manager agent
            max_rounds: Maximum number of debate rounds
        """
        self.bull_researcher = bull_researcher
        self.bear_researcher = bear_researcher
        self.research_manager = research_manager
        self.max_rounds = max_rounds

        logger.info("✅ Research Debate Orchestrator initialized with bull researcher, bear researcher and research manager")

    def _get_round_prompts(self, round_num: int) -> tuple[Msg, Msg]:
        """根据辩论轮次返回不同阶段的提示词。

        第1轮：充分陈述——完整表达初始观点，不急于反驳
        第2轮：对抗辩论——针对对方具体论点进行反驳交锋
        第3轮+：收敛精炼——承认对方合理点，聚焦核心分歧，减少偏差

        Args:
            round_num: 当前轮次 (1-based)

        Returns:
            (bull_prompt, bear_prompt) 元组
        """
        if round_num == 1:
            # 第1轮：全面陈述阶段
            bull_content = (
                "请看涨研究员全面阐述您的投资观点。\n"
                "这是第一轮陈述，请深入分析公司的增长潜力、竞争优势和积极指标，"
                "用充分的数据和事实构建您的看涨论证。\n"
                "无需急于反驳对方，重点是完整、客观地呈现您的分析基础。"
            )
            bear_content = (
                "请看跌研究员全面阐述您的投资观点。\n"
                "这是第一轮陈述，请深入分析公司的风险因素、竞争劣势和负面指标，"
                "用充分的数据和事实构建您的看跌论证。\n"
                "无需急于反驳对方，重点是完整、客观地呈现您的分析基础。"
            )
        elif round_num == 2:
            # 第2轮：对抗辩论阶段
            bull_content = (
                "请看涨研究员针对看跌研究员的观点进行反驳。\n"
                "仔细审视对方在前一轮中提出的风险担忧和负面论据：\n"
                "- 指出对方论证中的数据偏差、逻辑漏洞或过度悲观的假设\n"
                "- 用具体的市场数据、财务指标或行业趋势反驳对方的核心论点\n"
                "- 说明为什么您认为增长机会超过了对方提到的风险\n"
                "请聚焦于最关键的分歧点，用精准的论据展开交锋。"
            )
            bear_content = (
                "请看跌研究员针对看涨研究员的观点进行反驳。\n"
                "仔细审视对方在前一轮中提出的增长预期和积极论据：\n"
                "- 指出对方论证中的数据偏差、逻辑漏洞或过度乐观的假设\n"
                "- 用具体的市场数据、财务指标或行业风险反驳对方的核心论点\n"
                "- 说明为什么您认为风险因素超过了对方提到的机会\n"
                "请聚焦于最关键的分歧点，用精准的论据展开交锋。"
            )
        else:
            # 第3轮及以后：收敛与精炼阶段
            bull_content = (
                "请看涨研究员在综合前几轮辩论后，精炼您的核心观点。\n"
                "在这一轮中，请：\n"
                "- 承认看跌研究员提出的合理风险点（如果存在）\n"
                "- 说明即使考虑这些风险，为什么看涨立场仍然更有说服力\n"
                "- 聚焦于1-2个最强有力的看涨论据，避免重复已讨论的内容\n"
                "- 明确指出双方的核心分歧是什么，为什么您的判断更可靠\n"
                "目标是减少确认偏见，展示批判性思维，找到最有力的投资依据。"
            )
            bear_content = (
                "请看跌研究员在综合前几轮辩论后，精炼您的核心观点。\n"
                "在这一轮中，请：\n"
                "- 承认看涨研究员提出的合理机会点（如果存在）\n"
                "- 说明即使考虑这些机会，为什么看跌立场仍然更有说服力\n"
                "- 聚焦于1-2个最强有力的看跌论据，避免重复已讨论的内容\n"
                "- 明确指出双方的核心分歧是什么，为什么您的判断更可靠\n"
                "目标是减少确认偏见，展示批判性思维，找到最有力的风险警示。"
            )

        bull_prompt = Msg(name="DebateModerator", role="user", content=bull_content)
        bear_prompt = Msg(name="DebateModerator", role="user", content=bear_content)
        return bull_prompt, bear_prompt

    async def run_debate(
        self,
        company_name: str,
    ) -> tuple[str, Msg]:
        """Run the complete multi-agent debate and return the final decision.

        Args:
            company_name: Name of the company being analyzed

        Returns:
            Tuple of (debate_history, final_decision) from the research manager
        """
        logger.info(f"🚀 Starting research debate for {company_name}")

        # Use MsgHub for message broadcasting between researchers
        async with MsgHub(participants=[self.bull_researcher, self.bear_researcher, self.research_manager]):
            # Run debate rounds
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"🔄 Starting research debate round {round_num}")

                # 根据轮次获取不同阶段的提示词
                bull_prompt, bear_prompt = self._get_round_prompts(round_num)

                # Run all researchers concurrently within the MsgHub context
                await call_agent_with_retry(self.bull_researcher, bull_prompt)
                await call_agent_with_retry(self.bear_researcher, bear_prompt)

                logger.info(f"📝 Research debate round {round_num} completed")

        # Have research manager make final decision
        logger.info("⚖️ Requesting final decision from Research Manager")

        research_manager_prompt = Msg(
            name="DebateOrchestrator",
            role="user",
            content="""作为投资决策经理，请基于以下辩论历史做出最终的投资决策：

请给出明确的买入、卖出或持有建议，并说明理由。
请用中文撰写所有分析内容和建议，必须使用真实数据和事实提供决策支撑。
""",
        )

        final_decision = await call_agent_with_retry(self.research_manager, research_manager_prompt)
        logger.info("✅ Research debate completed")

        return final_decision


def create_research_debate_orchestrator(
    bull_researcher,
    bear_researcher,
    research_manager,
    max_rounds: int = 3,
) -> ResearchDebateOrchestrator:
    """Create a research debate orchestrator.

    Args:
        bull_researcher: Bull researcher agent
        bear_researcher: Bear researcher agent
        research_manager: Research manager agent
        max_rounds: Maximum number of debate rounds

    Returns:
        Configured ResearchDebateOrchestrator instance
    """
    return ResearchDebateOrchestrator(bull_researcher, bear_researcher, research_manager, max_rounds)
