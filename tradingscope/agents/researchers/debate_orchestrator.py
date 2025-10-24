"""Multi-Agent Debate Orchestrator for Bull/Bear Researchers using AgentScope."""

from __future__ import annotations

# Local imports
from agentscope import logger

# AgentScope imports
from agentscope.message import Msg
from agentscope.pipeline import MsgHub


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

        # Format prompts for each researcher
        bull_prompt = Msg(
            name="DebateModerator",
            role="user",
            content="""你作为看涨研究员，请基于所有研究报告和当前辩论历史，提出你的观点和论据。

请用中文回答，重点关注增长潜力、竞争优势和积极的市场指标。你需要：
1. 表达清晰的看涨观点
2. 使用数据和事实支撑你的论点
3. 直接回应和反驳看跌分析师的观点
4. 在多轮辩论中逐步深化你的论点
""",
        )

        bear_prompt = Msg(
            name="DebateModerator",
            role="user",
            content="""你作为看跌研究员，请基于所有研究报告、当前辩论历史和看涨研究员的最新观点，提出你的反驳观点和论据。

请用中文回答，重点关注风险和挑战、竞争劣势和负面指标。你需要：
1. 表达清晰的看跌观点
2. 使用数据和事实支撑你的论点
3. 直接回应和反驳看涨分析师的观点
4. 在多轮辩论中逐步深化你的论点
""",
        )

        # Use MsgHub for message broadcasting between researchers
        async with MsgHub(participants=[self.bull_researcher, self.bear_researcher, self.research_manager]):
            # Run debate rounds
            for round_num in range(1, self.max_rounds + 1):
                # Run this round
                logger.info(f"🔄 Starting research debate round {round_num}")

                # Run all researchers concurrently within the MsgHub context
                await self.bull_researcher(bull_prompt)
                await self.bear_researcher(bear_prompt)

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

        final_decision = await self.research_manager(research_manager_prompt)
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
