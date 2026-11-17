"""Multi-Agent Debate Orchestrator for Bull/Bear Researchers using AgentScope."""

from __future__ import annotations

from collections.abc import Sequence

from agentscope import logger
from agentscope.message import UserMsg

from tradingscope.agents.output import AgentOutputBase, ResearchManagerOutput
from tradingscope.agents.utils.agent_utils import build_debate_round_prompt, call_agent_with_retry


class ResearchDebateOrchestrator:
    """Orchestrates a multi-agent debate between bull and bear researchers."""

    def __init__(
        self,
        bull_researcher,
        bear_researcher,
        research_manager,
        structured_runner,
        max_rounds: int = 3,
        reference_outputs: Sequence[AgentOutputBase] = (),
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
        self.structured_runner = structured_runner
        self.max_rounds = max_rounds
        self.reference_outputs = tuple(reference_outputs)

        logger.info("✅ Research Debate Orchestrator initialized with bull researcher, bear researcher and research manager")

    def _get_round_prompts(self, round_num: int) -> tuple[UserMsg, UserMsg]:
        """根据辩论轮次返回提示词：首轮陈述，中间轮反驳，最后一轮收敛。"""
        bull_content = build_debate_round_prompt(
            round_num,
            self.max_rounds,
            opening="请看涨研究员陈述未来 1-5 个交易日的看涨观点，聚焦增长潜力、竞争优势和积极指标中最关键的证据。",
            rebuttal="请看涨研究员反驳看跌研究员已发表的观点，指出其数据偏差、逻辑漏洞或过度悲观的假设。",
        )
        bear_content = build_debate_round_prompt(
            round_num,
            self.max_rounds,
            opening="请看跌研究员陈述未来 1-5 个交易日的看跌观点，聚焦风险因素、竞争劣势和负面指标中最关键的证据。",
            rebuttal="请看跌研究员反驳看涨研究员已发表的观点，指出其数据偏差、逻辑漏洞或过度乐观的假设。",
        )

        bull_prompt = UserMsg(name="DebateModerator", content=bull_content)
        bear_prompt = UserMsg(name="DebateModerator", content=bear_content)
        return bull_prompt, bear_prompt

    async def run_debate(
        self,
        ticker: str,
    ):
        """Run the complete multi-agent debate and return the final decision.

        Args:
            ticker: Stock ticker being analyzed

        Returns:
            Final decision Msg from the research manager
        """
        logger.info(f"🚀 Starting research debate for {ticker}")

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"🔄 Starting research debate round {round_num}")

            bull_prompt, bear_prompt = self._get_round_prompts(round_num)

            # Bull researcher speaks
            bull_response = await call_agent_with_retry(self.bull_researcher, bull_prompt)
            # Broadcast to bear and research manager
            await self.bear_researcher.observe(bull_response)
            await self.research_manager.observe(bull_response)

            # Bear researcher speaks
            bear_response = await call_agent_with_retry(self.bear_researcher, bear_prompt)
            # Broadcast to bull and research manager
            await self.bull_researcher.observe(bear_response)
            await self.research_manager.observe(bear_response)

            logger.info(f"📝 Research debate round {round_num} completed")

        # Have research manager make final decision
        logger.info("⚖️ Requesting final decision from Research Manager")

        research_manager_prompt = UserMsg(
            name="DebateOrchestrator",
            content="""作为投资决策经理，请基于以下辩论历史做出最终的投资决策：

请给出明确的买入、卖出或持有建议，并说明理由。
请用中文撰写所有分析内容和建议，必须使用真实数据和事实提供决策支撑。
""",
        )

        final_decision = await self.structured_runner.run(
            self.research_manager,
            ResearchManagerOutput,
            research_manager_prompt,
            reference_outputs=self.reference_outputs,
        )
        logger.info("✅ Research debate completed")

        return final_decision


def create_research_debate_orchestrator(
    bull_researcher,
    bear_researcher,
    research_manager,
    structured_runner,
    max_rounds: int = 3,
    reference_outputs: Sequence[AgentOutputBase] = (),
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
    return ResearchDebateOrchestrator(
        bull_researcher,
        bear_researcher,
        research_manager,
        structured_runner,
        max_rounds=max_rounds,
        reference_outputs=reference_outputs,
    )
