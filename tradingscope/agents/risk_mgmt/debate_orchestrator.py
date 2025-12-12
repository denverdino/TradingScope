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
            content="""请激进风险分析师发表您的观点""",
        )

        conservative_prompt = Msg(
            name="DebateOrchestrator",
            role="user",
            content="""请安全/保守风险分析师发表您的观点""",
        )

        neutral_prompt = Msg(
            name="DebateOrchestrator",
            role="user",
            content="""请中性风险分析师发表您的观点""",
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

        final_decision = await self.risk_manager(risk_manager_prompt)

        logger.info("✅ Risk management debate completed")

        return final_decision


def create_debate_orchestrator(
    aggressive_agent,
    conservative_agent,
    neutral_agent,
    risk_manager,
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
        risk_manager=risk_manager,
        max_rounds=max_rounds,
    )
