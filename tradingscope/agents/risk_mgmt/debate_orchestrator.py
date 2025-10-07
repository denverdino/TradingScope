"""Multi-Agent Debate Orchestrator for Risk Management Team using AgentScope."""

from __future__ import annotations

# AgentScope imports
from agentscope.message import Msg
from agentscope.pipeline import MsgHub

# Local imports
from tradingscope.utils.logging_init import get_logger

logger = get_logger("default")


class RiskDebateOrchestrator:
    """Orchestrates a multi-agent debate between risk management team members."""

    def __init__(
        self,
        aggressive_agent,
        conservative_agent,
        neutral_agent,
        risk_manager,
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

        # Create the risk manager agent
        self.risk_manager = risk_manager

        logger.info("✅ Risk Debate Orchestrator initialized with 3 debators and 1 risk manager")

    async def run_debate(
        self,
        company_name: str,
        trader_plan: str,
    ) -> Msg:
        """Run the complete multi-agent debate and return the final decision.

        Args:
            company_name: Name of the company being analyzed
            trader_plan: Trader's investment plan

        Returns:
            Final decision from the risk manager
        """
        logger.info(f"🚀 Starting risk management debate for {company_name}")

        # Format prompts for each researcher
        aggressive_prompt = Msg(
            name="DebateOrchestrator",
            role="user",
            content="""作为激进风险分析师，您的职责是积极倡导高回报、高风险的投资机会，强调大胆策略和竞争优势。在评估交易员的决策或计划时，请重点关注潜在的上涨空间、增长潜力和创新收益——即使这些伴随着较高的风险。使用提供的市场数据和情绪分析来加强您的论点，并挑战对立观点。具体来说，请直接回应保守和中性分析师提出的每个观点，用数据驱动的反驳和有说服力的推理进行反击。突出他们的谨慎态度可能错过的关键机会，或者他们的假设可能过于保守的地方。""",
        )

        conservative_prompt = Msg(
            name="DebateOrchestrator",
            role="user",
            content="""作为安全/保守风险分析师，您的主要目标是保护资产、最小化波动性，并确保稳定、可靠的增长。您优先考虑稳定性、安全性和风险缓解，仔细评估潜在损失、经济衰退和市场波动。在评估交易员的决策或计划时，请批判性地审查高风险要素，指出决策可能使公司面临不当风险的地方，以及更谨慎的替代方案如何能够确保长期收益。""",
        )

        neutral_prompt = Msg(
            name="DebateOrchestrator",
            role="user",
            content="""作为中性风险分析师，您的角色是提供平衡的视角，权衡交易员决策或计划的潜在收益和风险。您优先考虑全面的方法，评估上行和下行风险，同时考虑更广泛的市场趋势、潜在的经济变化和多元化策略。""",
        )

        # Use MsgHub for message broadcasting between debators
        async with MsgHub(participants=[self.aggressive_agent, self.conservative_agent, self.neutral_agent, self.risk_manager]):
            # Run debate rounds
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"🔄 Starting risk management debate round {round_num}")

                # Run all debators concurrently within the MsgHub context
                await self.aggressive_agent(aggressive_prompt)
                await self.conservative_agent(conservative_prompt)
                await self.neutral_agent(neutral_prompt)

                logger.info(f"📝 Risk management debate round {round_num} completed")

        # Have risk manager make final decision
        logger.info("⚖️ Requesting final decision from Risk Manager")

        risk_manager_prompt = Msg(
            name="DebateOrchestrator",
            role="user",
            content=f"""作为风险管理委员会主席和辩论主持人，您的目标是评估三位风险分析师——激进、中性和安全/保守——之间的辩论，并确定交易员的最佳行动方案。您的决策必须产生明确的建议：买入、卖出或持有。只有在有具体论据强烈支持时才选择持有，而不是在所有方面都似乎有效时作为后备选择。力求清晰和果断。

决策指导原则：
1. **总结关键论点**：提取每位分析师的最强观点，重点关注与背景的相关性。
2. **提供理由**：用辩论中的直接引用和反驳论点支持您的建议。
3. **完善交易员计划**：从交易员的原始计划**{trader_plan}**开始，根据分析师的见解进行调整。
4. **从过去的错误中学习**：使用过去的经验教训来解决先前的误判，改进您现在做出的决策，确保您不会做出错误的买入/卖出/持有决定而亏损。

交付成果：
- 明确且可操作的建议：买入、卖出或持有。
- 基于辩论和过去反思的详细推理。


专注于可操作的见解和持续改进。建立在过去经验教训的基础上，批判性地评估所有观点，确保每个决策都能带来更好的结果。请用中文撰写所有分析内容和建议。""",
        )

        final_decision = await self.risk_manager(risk_manager_prompt)

        logger.info("✅ Risk management debate completed")

        return final_decision


def create_debate_orchestrator(
    aggressive_agent,
    conservative_agent,
    neutral_agent,
    risk_manager,
    max_rounds: int = 1,
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
        risk_manager=risk_manager,
        max_rounds=max_rounds,
    )
