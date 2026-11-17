"""Multi-Agent Debate Orchestrator for Risk Management Team using AgentScope."""

from __future__ import annotations

from collections.abc import Sequence

from agentscope import logger
from agentscope.message import UserMsg

from tradingscope.agents.output import AgentOutputBase, PortfolioManagerOutput
from tradingscope.agents.utils.agent_utils import build_debate_round_prompt, call_agent_with_retry


class RiskDebateOrchestrator:
    """Orchestrates a multi-agent debate between risk management team members."""

    def __init__(
        self,
        aggressive_agent,
        conservative_agent,
        neutral_agent,
        portfolio_manager,
        structured_runner,
        max_rounds: int,
        reference_outputs: Sequence[AgentOutputBase] = (),
    ):
        """Initialize the debate orchestrator.

        Args:
            aggressive_agent: Aggressive debater agent
            conservative_agent: Conservative debater agent
            neutral_agent: Neutral debater agent
            portfolio_manager: Portfolio manager agent
            max_rounds: Maximum number of debate rounds
        """
        self.max_rounds = max_rounds

        # Create the debater agents
        self.aggressive_agent = aggressive_agent
        self.conservative_agent = conservative_agent
        self.neutral_agent = neutral_agent

        # Create the portfolio manager agent
        self.portfolio_manager = portfolio_manager
        self.structured_runner = structured_runner
        self.reference_outputs = tuple(reference_outputs)

        logger.info("✅ Risk Debate Orchestrator initialized with 3 debaters and 1 portfolio manager")

    def _get_round_prompts(self, round_num: int) -> tuple[UserMsg, UserMsg, UserMsg]:
        """根据辩论轮次返回提示词：首轮陈述，中间轮反驳，最后一轮收敛。"""
        aggressive_content = build_debate_round_prompt(
            round_num,
            self.max_rounds,
            opening="请激进风险分析师评估交易计划，聚焦高回报潜力、增长机会和竞争优势中最关键的证据。",
            rebuttal="请激进风险分析师反驳保守和中性分析师已发表的观点，指出其过度规避风险或可能错失的机会。",
        )
        conservative_content = build_debate_round_prompt(
            round_num,
            self.max_rounds,
            opening="请安全/保守风险分析师评估交易计划，聚焦潜在风险、波动性和下行威胁中最关键的证据。",
            rebuttal="请安全/保守风险分析师反驳激进和中性分析师已发表的观点，指出其过度乐观或低估的下行风险。",
        )
        neutral_content = build_debate_round_prompt(
            round_num,
            self.max_rounds,
            opening="请中性风险分析师评估交易计划，聚焦风险收益平衡和市场趋势中最关键的证据。",
            rebuttal="请中性风险分析师反驳激进和保守分析师已发表的观点，指出其过度冒险或过度谨慎之处。",
        )

        aggressive_prompt = UserMsg(name="DebateOrchestrator", content=aggressive_content)
        conservative_prompt = UserMsg(name="DebateOrchestrator", content=conservative_content)
        neutral_prompt = UserMsg(name="DebateOrchestrator", content=neutral_content)
        return aggressive_prompt, conservative_prompt, neutral_prompt

    async def run_debate(
        self,
        ticker: str,
    ):
        """Run the complete multi-agent debate and return the final decision.

        Args:
            ticker: Stock ticker being analyzed

        Returns:
            Final decision Msg from the portfolio manager
        """
        logger.info(f"🚀 Starting risk management debate for {ticker}")

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"🔄 Starting risk management debate round {round_num}")

            aggressive_prompt, conservative_prompt, neutral_prompt = self._get_round_prompts(round_num)

            # Aggressive debater speaks
            agg_response = await call_agent_with_retry(self.aggressive_agent, aggressive_prompt)
            await self.conservative_agent.observe(agg_response)
            await self.neutral_agent.observe(agg_response)
            await self.portfolio_manager.observe(agg_response)

            # Conservative debater speaks
            con_response = await call_agent_with_retry(self.conservative_agent, conservative_prompt)
            await self.aggressive_agent.observe(con_response)
            await self.neutral_agent.observe(con_response)
            await self.portfolio_manager.observe(con_response)

            # Neutral debater speaks
            neu_response = await call_agent_with_retry(self.neutral_agent, neutral_prompt)
            await self.aggressive_agent.observe(neu_response)
            await self.conservative_agent.observe(neu_response)
            await self.portfolio_manager.observe(neu_response)

            logger.info(f"📝 Risk management debate round {round_num} completed")

        # Have portfolio manager make final decision
        logger.info("⚖️ Requesting final decision from Portfolio Manager")

        portfolio_manager_prompt = UserMsg(
            name="DebateOrchestrator",
            content="""请基于以上三位风险分析师的辩论，按系统指令给出最终风险调整决策素材：明确买入、卖出或持有。
只有在有具体论据强烈支持时才选择持有，而不是在所有方面都似乎有效时作为后备选择。请用中文撰写。
""",
        )

        final_decision = await self.structured_runner.run(
            self.portfolio_manager,
            PortfolioManagerOutput,
            portfolio_manager_prompt,
            reference_outputs=self.reference_outputs,
        )

        logger.info("✅ Risk management debate completed")

        return final_decision


def create_risk_debate_orchestrator(
    aggressive_agent,
    conservative_agent,
    neutral_agent,
    portfolio_manager,
    structured_runner,
    max_rounds: int = 3,
    reference_outputs: Sequence[AgentOutputBase] = (),
) -> RiskDebateOrchestrator:
    """Create a risk management debate orchestrator.

    Args:
        aggressive_agent: Aggressive debater agent
        conservative_agent: Conservative debater agent
        neutral_agent: Neutral debater agent
        portfolio_manager: Portfolio manager agent
        max_rounds: Maximum number of debate rounds

    Returns:
        Configured RiskDebateOrchestrator instance
    """
    return RiskDebateOrchestrator(
        aggressive_agent=aggressive_agent,
        conservative_agent=conservative_agent,
        neutral_agent=neutral_agent,
        portfolio_manager=portfolio_manager,
        structured_runner=structured_runner,
        max_rounds=max_rounds,
        reference_outputs=reference_outputs,
    )
