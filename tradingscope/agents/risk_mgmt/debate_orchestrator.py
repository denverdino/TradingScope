"""Multi-Agent Debate Orchestrator for Risk Management Team using AgentScope."""

from __future__ import annotations

# Local imports
from agentscope import logger

# AgentScope imports
from agentscope.message import Msg
from agentscope.pipeline import MsgHub


class RiskDebateOrchestrator:
    """Orchestrates a multi-agent debate between risk management team members."""

    def __init__(
        self,
        aggressive_agent,
        conservative_agent,
        neutral_agent,
        portfolio_manager,
        max_rounds: int,
    ):
        """Initialize the debate orchestrator.

        Args:
            model_config: Configuration for the language model
            max_rounds: Maximum number of debate rounds
        """
        self.max_rounds = max_rounds

        # Create the debator agents
        self.aggressive_agent = aggressive_agent
        self.conservative_agent = conservative_agent
        self.neutral_agent = neutral_agent

        # Create the portfolio manager agent
        self.portfolio_manager = portfolio_manager

        logger.info("✅ Risk Debate Orchestrator initialized with 3 debators and 1 portfolio manager")

    def _get_round_prompts(self, round_num: int) -> tuple[Msg, Msg, Msg]:
        """根据辩论轮次返回不同阶段的提示词。

        第1轮：充分陈述——完整表达初始观点，不急于反驳
        第2轮：对抗辩论——针对对方具体论点进行反驳交锋
        第3轮+：收敛精炼——承认对方合理点，聚焦核心分歧，减少偏差

        Args:
            round_num: 当前轮次 (1-based)

        Returns:
            (aggressive_prompt, conservative_prompt, neutral_prompt) 元组
        """
        if round_num == 1:
            # 第1轮：全面陈述阶段
            aggressive_content = (
                "请激进风险分析师全面阐述您对交易计划的风险评估。\n"
                "这是第一轮陈述，请深入分析计划的高回报潜力、增长机会和竞争优势，"
                "用市场数据和情绪分析构建您的论证。\n"
                "无需急于反驳其他分析师，重点是完整、客观地呈现您的激进视角基础。"
            )
            conservative_content = (
                "请安全/保守风险分析师全面阐述您对交易计划的风险评估。\n"
                "这是第一轮陈述，请深入分析计划的潜在风险、稳定性要求和下行威胁，"
                "用风险数据和历史波动性构建您的论证。\n"
                "无需急于反驳其他分析师，重点是完整、客观地呈现您的保守视角基础。"
            )
            neutral_content = (
                "请中性风险分析师全面阐述您对交易计划的平衡评估。\n"
                "这是第一轮陈述，请深入分析计划的风险收益平衡、市场趋势和多元化考量，"
                "用全面的数据构建您的论证。\n"
                "无需急于反驳其他分析师，重点是完整、客观地呈现您的中性视角基础。"
            )
        elif round_num == 2:
            # 第2轮：对抗辩论阶段
            aggressive_content = (
                "请激进风险分析师针对保守和中性分析师的观点进行反驳。\n"
                "仔细审视他们在前一轮中提出的风险担忧和谨慎建议：\n"
                "- 指出保守分析师的论证中过度规避风险、可能错失的重大机会\n"
                "- 指出中性分析师的平衡立场可能导致的平庸收益\n"
                "- 用具体的市场数据、成功案例说明为什么高风险高回报策略更优\n"
                "请聚焦于最关键的分歧点，展示激进策略的优势。"
            )
            conservative_content = (
                "请安全/保守风险分析师针对激进和中性分析师的观点进行反驳。\n"
                "仔细审视他们在前一轮中提出的收益预期和策略建议：\n"
                "- 指出激进分析师的论证中过度乐观、忽视的重大风险点\n"
                "- 指出中性分析师的平衡立场可能低估的下行威胁\n"
                "- 用具体的风险案例、历史数据说明为什么保守策略更安全\n"
                "请聚焦于最关键的分歧点，展示保守策略的必要性。"
            )
            neutral_content = (
                "请中性风险分析师针对激进和保守分析师的观点进行反驳。\n"
                "仔细审视他们在前一轮中提出的极端立场：\n"
                "- 指出激进分析师论证中的过度冒险和潜在盲点\n"
                "- 指出保守分析师论证中的过度谨慎和错失机会\n"
                "- 用数据说明为什么平衡的中性策略能够实现稳健增长同时控制风险\n"
                "请聚焦于最关键的分歧点，展示平衡策略的合理性。"
            )
        else:
            # 第3轮及以后：收敛与精炼阶段
            aggressive_content = (
                "请激进风险分析师在综合前几轮辩论后，精炼您的核心观点。\n"
                "在这一轮中，请：\n"
                "- 承认保守和中性分析师提出的合理风险点（如果存在）\n"
                "- 说明即使考虑这些风险，为什么激进策略仍然更适合当前市场环境\n"
                "- 聚焦于1-2个最强有力的高回报论据，避免重复已讨论的内容\n"
                "- 明确指出三方的核心分歧是什么，为什么您的风险承受判断更合理\n"
                "目标是减少确认偏见，展示批判性思维，找到最优风险收益平衡点。"
            )
            conservative_content = (
                "请安全/保守风险分析师在综合前几轮辩论后，精炼您的核心观点。\n"
                "在这一轮中，请：\n"
                "- 承认激进和中性分析师提出的合理机会点（如果存在）\n"
                "- 说明即使考虑这些机会，为什么保守策略仍然更符合稳健投资原则\n"
                "- 聚焦于1-2个最强有力的风险警示论据，避免重复已讨论的内容\n"
                "- 明确指出三方的核心分歧是什么，为什么您的风险控制判断更可靠\n"
                "目标是减少确认偏见，展示批判性思维，找到最优风险防御策略。"
            )
            neutral_content = (
                "请中性风险分析师在综合前几轮辩论后，精炼您的核心观点。\n"
                "在这一轮中，请：\n"
                "- 承认激进和保守分析师各自提出的合理点（如果存在）\n"
                "- 说明为什么平衡的中性策略能够整合双方优点，避免极端风险\n"
                "- 聚焦于1-2个最强有力的平衡策略论据，避免重复已讨论的内容\n"
                "- 明确指出三方的核心分歧是什么，为什么中性立场最符合理性投资\n"
                "目标是减少确认偏见，展示批判性思维，找到最优风险收益平衡点。"
            )

        aggressive_prompt = Msg(name="DebateOrchestrator", role="user", content=aggressive_content)
        conservative_prompt = Msg(name="DebateOrchestrator", role="user", content=conservative_content)
        neutral_prompt = Msg(name="DebateOrchestrator", role="user", content=neutral_content)
        return aggressive_prompt, conservative_prompt, neutral_prompt

    async def run_debate(
        self,
        company_name: str,
    ) -> Msg:
        """Run the complete multi-agent debate and return the final decision.

        Args:
            company_name: Name of the company being analyzed

        Returns:
            Final decision from the portfolio manager
        """
        logger.info(f"🚀 Starting risk management debate for {company_name}")

        # Use MsgHub for message broadcasting between debators
        async with MsgHub(participants=[self.aggressive_agent, self.conservative_agent, self.neutral_agent, self.portfolio_manager]):
            # Run debate rounds
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"🔄 Starting risk management debate round {round_num}")

                # 根据轮次获取不同阶段的提示词
                aggressive_prompt, conservative_prompt, neutral_prompt = self._get_round_prompts(round_num)

                # Run all debators concurrently within the MsgHub context
                await self.aggressive_agent(aggressive_prompt)
                await self.conservative_agent(conservative_prompt)
                await self.neutral_agent(neutral_prompt)

                logger.info(f"📝 Risk management debate round {round_num} completed")

        # Have portfolio manager make final decision
        logger.info("⚖️ Requesting final decision from Portfolio Manager")

        portfolio_manager_prompt = Msg(
            name="DebateOrchestrator",
            role="user",
            content="""作为风险管理委员会主席，您的目标是评估三位风险分析师——激进、中性和安全/保守——之间的辩论，并确定交易员的最佳行动方案。您的决策必须产生明确的建议：买入、卖出或持有。只有在有具体论据强烈支持时才选择持有，而不是在所有方面都似乎有效时作为后备选择。力求清晰和果断。

决策指导原则：
1. **总结关键论点**：提取每位分析师的最强观点，重点关注与背景的相关性。
2. **提供理由**：用辩论中的直接引用和反驳论点支持您的建议。
3. **完善交易员计划**：从交易员的交易计划开始，根据分析师的见解进行调整。
4. **从过去的错误中学习**：使用过去的经验教训来解决先前的误判，改进您现在做出的决策，确保您不会做出错误的买入/卖出/持有决定而亏损。

交付成果：
- 明确且可操作的建议：买入、卖出或持有。
- 基于辩论和过去反思的详细推理。


专注于可操作的见解和持续改进。建立在过去经验教训的基础上，批判性地评估所有观点，确保每个决策都能带来更好的结果。请用中文撰写所有分析内容和建议。
""",
        )

        final_decision = await self.portfolio_manager(portfolio_manager_prompt)

        logger.info("✅ Risk management debate completed")

        return final_decision


def create_debate_orchestrator(
    aggressive_agent,
    conservative_agent,
    neutral_agent,
    portfolio_manager,
    max_rounds: int = 3,
) -> RiskDebateOrchestrator:
    """Create a risk management debate orchestrator.

    Args:
        model: The language model to use
        max_rounds: Maximum number of debate rounds

    Returns:
        Configured RiskDebateOrchestrator instance
    """
    return RiskDebateOrchestrator(
        aggressive_agent=aggressive_agent,
        conservative_agent=conservative_agent,
        neutral_agent=neutral_agent,
        portfolio_manager=portfolio_manager,
        max_rounds=max_rounds,
    )
