"""Structured output models for TradingScope agents.

Defines Pydantic models for each analysis stage, enabling:
1. AgentScope native structured output via the `structured_model` parameter
2. JSON serialization for downstream system consumption
3. Minimal regex fallback when structured output is unavailable
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tradingscope.agents.utils.context import AgentContext

# --- Analyst structured output models ---


class AnalystStructuredOutput(BaseModel):
    """Base structured output model shared by all analyst agents."""

    direction: Literal["bullish", "bearish", "neutral"] = "neutral"
    action: Literal["buy", "sell", "hold"] = "hold"
    confidence: float = Field(default=0.5, ge=0, le=1, description="分析置信度，0-1之间")
    entry_price: Optional[float] = Field(default=None, description="建议入场价")
    target_price: Optional[float] = Field(default=None, description="目标价位")
    stop_loss: Optional[float] = Field(default=None, description="止损价位")
    reasoning: str = Field(default="", description="分析核心理由")


class MarketAnalystStructuredOutput(AnalystStructuredOutput):
    """Structured output for the Market (technical) analyst."""

    signal_strength: str = Field(default="", description="信号强度：强/中等/弱")
    invalidation_conditions: str = Field(default="", description="交易计划失效条件")


class FundamentalsAnalystStructuredOutput(AnalystStructuredOutput):
    """Structured output for the Fundamentals analyst."""

    valuation_assessment: str = Field(default="", description="估值评估：偏高/合理/偏低")
    key_catalysts: List[str] = Field(default_factory=list, description="短期催化剂列表")
    key_risks: List[str] = Field(default_factory=list, description="短期风险列表")


class NewsAnalystStructuredOutput(AnalystStructuredOutput):
    """Structured output for the News analyst."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(default="neutral", description="新闻情绪方向")
    key_events: List[str] = Field(default_factory=list, description="关键新闻事件列表")


class SocialMediaAnalystStructuredOutput(AnalystStructuredOutput):
    """Structured output for the Social Media analyst."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(default="neutral", description="社交媒体情绪方向")
    sentiment_score: Optional[float] = Field(default=None, ge=1, le=10, description="情绪评分，1-10")


# --- Research Manager structured output model ---


class ResearchManagerStructuredOutput(BaseModel):
    """Structured output for the Research Manager agent."""

    direction: Literal["bullish", "bearish", "neutral"] = "neutral"
    action: Literal["buy", "sell", "hold"] = "hold"
    confidence: float = Field(default=0.5, ge=0, le=1, description="投资决策置信度，0-1之间")
    entry_price: Optional[float] = Field(default=None, description="建议入场价")
    target_price: Optional[float] = Field(default=None, description="目标价位")
    stop_loss: Optional[float] = Field(default=None, description="止损价位")
    bull_viewpoints: str = Field(default="", description="看涨论点摘要")
    bear_viewpoints: str = Field(default="", description="看跌论点摘要")
    adopted_reasoning: str = Field(default="", description="决策采纳理由")
    reasoning: str = Field(default="", description="投资决策核心理由")


# --- Models used as AgentScope structured_model parameter ---


class PredictionData(BaseModel):
    """Core trading prediction data — used by both Trader and Portfolio agents."""

    direction: Literal["bullish", "bearish", "neutral"] = "neutral"
    action: Literal["buy", "sell", "hold"] = "hold"
    confidence: float = Field(default=0.5, ge=0, le=1)
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""


class TraderStructuredOutput(BaseModel):
    """Structured output model for the Trader agent.

    Passed as `structured_model` to ReActAgent.reply() so the LLM
    fills these fields directly via the generate_response tool call.
    """

    direction: Literal["bullish", "bearish", "neutral"] = "neutral"
    action: Literal["buy", "sell", "hold"] = "hold"
    confidence: float = Field(default=0.5, ge=0, le=1, description="交易置信度，0-1之间")
    entry_price: Optional[float] = Field(default=None, description="建议入场价")
    target_price: Optional[float] = Field(default=None, description="目标价位")
    stop_loss: Optional[float] = Field(default=None, description="止损价位")
    risk_reward_ratio: Optional[float] = Field(default=None, description="盈亏比，必须>=2:1")
    position_advice: str = Field(default="", description="仓位建议：轻仓/中等仓位/重仓")
    risk_score: Optional[float] = Field(default=None, ge=0, le=1, description="风险评分，0-1之间")
    time_stop_days: Optional[int] = Field(default=None, description="时间止损天数")
    invalidation_conditions: str = Field(default="", description="交易计划失效条件")
    reasoning: str = Field(default="", description="交易决策核心理由")


class PortfolioStructuredOutput(BaseModel):
    """Structured output model for the Portfolio Manager agent.

    Passed as `structured_model` to ReActAgent.reply() so the LLM
    fills these fields directly via the generate_response tool call.
    """

    direction: Literal["bullish", "bearish", "neutral"] = "neutral"
    action: Literal["buy", "sell", "hold"] = "hold"
    confidence: float = Field(default=0.5, ge=0, le=1, description="最终交易置信度，0-1之间")
    entry_price: Optional[float] = Field(default=None, description="优化后入场价")
    target_price: Optional[float] = Field(default=None, description="优化后目标价")
    stop_loss: Optional[float] = Field(default=None, description="优化后止损价")
    position_advice: str = Field(default="", description="仓位建议：轻仓/中等仓位/重仓")
    risk_score: Optional[float] = Field(default=None, ge=0, le=1, description="风险评分，0-1之间")
    aggressive_viewpoint: str = Field(default="", description="激进派观点摘要")
    conservative_viewpoint: str = Field(default="", description="保守派观点摘要")
    neutral_viewpoint: str = Field(default="", description="中性派观点摘要")
    adopted_reasoning: str = Field(default="", description="最终决策采纳理由")
    risk_control_measures: List[str] = Field(default_factory=list, description="风险控制措施列表")
    invalidation_conditions: str = Field(default="", description="交易计划失效条件")
    reasoning: str = Field(default="", description="最终决策核心理由")


# --- Container models for the full analysis result ---


class AnalystDecision(BaseModel):
    """Structured decision from a single analyst agent."""

    prediction: PredictionData = Field(default_factory=PredictionData)
    # Domain-specific fields stored as-is from structured output
    extra: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_structured_output(cls, data: Dict[str, Any]) -> "AnalystDecision":
        """Build from AgentScope structured_output metadata."""
        pred = PredictionData(
            direction=data.get("direction", "neutral"),
            action=data.get("action", "hold"),
            confidence=data.get("confidence", 0.5),
            entry_price=data.get("entry_price"),
            target_price=data.get("target_price"),
            stop_loss=data.get("stop_loss"),
            reasoning=data.get("reasoning", ""),
        )
        # Collect remaining fields as extra
        known_pred_keys = {"direction", "action", "confidence", "entry_price", "target_price", "stop_loss", "reasoning"}
        extra = {k: v for k, v in data.items() if k not in known_pred_keys and v is not None and v != "" and v != []}
        return cls(prediction=pred, extra=extra)


class ResearchDecision(BaseModel):
    """Structured decision from the Research Manager."""

    prediction: PredictionData = Field(default_factory=PredictionData)
    bull_viewpoints: str = ""
    bear_viewpoints: str = ""
    adopted_reasoning: str = ""

    @classmethod
    def from_structured_output(cls, data: Dict[str, Any]) -> "ResearchDecision":
        """Build from AgentScope structured_output metadata."""
        return cls(
            prediction=PredictionData(
                direction=data.get("direction", "neutral"),
                action=data.get("action", "hold"),
                confidence=data.get("confidence", 0.5),
                entry_price=data.get("entry_price"),
                target_price=data.get("target_price"),
                stop_loss=data.get("stop_loss"),
                reasoning=data.get("reasoning", ""),
            ),
            bull_viewpoints=data.get("bull_viewpoints", ""),
            bear_viewpoints=data.get("bear_viewpoints", ""),
            adopted_reasoning=data.get("adopted_reasoning", ""),
        )


class AnalystReports(BaseModel):
    """Structured collection of all analyst reports."""

    market: str = ""
    fundamentals: str = ""
    news: str = ""
    social_media: str = ""
    market_structured: Optional[Dict[str, Any]] = None
    fundamentals_structured: Optional[Dict[str, Any]] = None
    news_structured: Optional[Dict[str, Any]] = None
    social_media_structured: Optional[Dict[str, Any]] = None


class TraderDecision(BaseModel):
    """Trader decision with risk parameters.

    Built from TraderStructuredOutput (AgentScope metadata)
    or regex fallback.
    """

    prediction: PredictionData = Field(default_factory=PredictionData)
    position_advice: str = ""
    risk_score: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    time_stop_days: Optional[int] = None
    invalidation_conditions: str = ""

    @classmethod
    def from_structured_output(cls, data: Dict[str, Any]) -> "TraderDecision":
        """Build from AgentScope structured_output metadata."""
        return cls(
            prediction=PredictionData(
                direction=data.get("direction", "neutral"),
                action=data.get("action", "hold"),
                confidence=data.get("confidence", 0.5),
                entry_price=data.get("entry_price"),
                target_price=data.get("target_price"),
                stop_loss=data.get("stop_loss"),
                reasoning=data.get("reasoning", ""),
            ),
            position_advice=data.get("position_advice", ""),
            risk_score=data.get("risk_score"),
            risk_reward_ratio=data.get("risk_reward_ratio"),
            time_stop_days=data.get("time_stop_days"),
            invalidation_conditions=data.get("invalidation_conditions", ""),
        )


class PortfolioDecision(BaseModel):
    """Final portfolio manager decision with risk parameters."""

    prediction: PredictionData = Field(default_factory=PredictionData)
    viewpoints_aggressive: str = ""
    viewpoints_conservative: str = ""
    viewpoints_neutral: str = ""
    adopted_reasoning: str = ""
    position_advice: str = ""
    risk_score: Optional[float] = None
    risk_control_measures: List[str] = Field(default_factory=list)
    invalidation_conditions: str = ""

    @classmethod
    def from_structured_output(cls, data: Dict[str, Any]) -> "PortfolioDecision":
        """Build from AgentScope structured_output metadata."""
        return cls(
            prediction=PredictionData(
                direction=data.get("direction", "neutral"),
                action=data.get("action", "hold"),
                confidence=data.get("confidence", 0.5),
                entry_price=data.get("entry_price"),
                target_price=data.get("target_price"),
                stop_loss=data.get("stop_loss"),
                reasoning=data.get("reasoning", ""),
            ),
            viewpoints_aggressive=data.get("aggressive_viewpoint", ""),
            viewpoints_conservative=data.get("conservative_viewpoint", ""),
            viewpoints_neutral=data.get("neutral_viewpoint", ""),
            adopted_reasoning=data.get("adopted_reasoning", ""),
            position_advice=data.get("position_advice", ""),
            risk_score=data.get("risk_score"),
            risk_control_measures=data.get("risk_control_measures", []),
            invalidation_conditions=data.get("invalidation_conditions", ""),
        )


class AnalysisResult(BaseModel):
    """Top-level structured analysis result combining all stages.

    This is the complete JSON-serializable output that downstream systems
    can consume directly alongside the Markdown report.
    """

    ticker: str = ""
    trade_date: str = ""
    latest_trading_date: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    analyst_reports: AnalystReports = Field(default_factory=AnalystReports)
    analyst_decisions: Dict[str, Any] = Field(default_factory=dict)
    researcher_investment_plan: str = ""
    research_decision: ResearchDecision = Field(default_factory=ResearchDecision)
    trader_decision: TraderDecision = Field(default_factory=TraderDecision)
    portfolio_decision: PortfolioDecision = Field(default_factory=PortfolioDecision)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_context(
        cls,
        context: AgentContext,
        trader_structured: Optional[Dict[str, Any]] = None,
        portfolio_structured: Optional[Dict[str, Any]] = None,
        research_structured: Optional[Dict[str, Any]] = None,
        market_structured: Optional[Dict[str, Any]] = None,
        fundamentals_structured: Optional[Dict[str, Any]] = None,
        news_structured: Optional[Dict[str, Any]] = None,
        social_media_structured: Optional[Dict[str, Any]] = None,
    ) -> "AnalysisResult":
        """Build an AnalysisResult from a populated AgentContext.

        Uses AgentScope structured_output metadata when available,
        falling back to regex-based extraction from Markdown text.
        """
        # Build trader decision
        if trader_structured:
            trader_decision = TraderDecision.from_structured_output(trader_structured)
        else:
            trader_pred_data = context.extract_prediction_data(
                source=context.trader_investment_plan,
            )
            trader_decision = TraderDecision(
                prediction=PredictionData(**trader_pred_data),
            )

        # Build portfolio decision
        if portfolio_structured:
            portfolio_decision = PortfolioDecision.from_structured_output(portfolio_structured)
        else:
            portfolio_pred_data = context.extract_prediction_data(
                source=context.final_trade_decision,
            )
            portfolio_decision = PortfolioDecision(
                prediction=PredictionData(**portfolio_pred_data),
            )

        # Build research decision
        if research_structured:
            research_decision = ResearchDecision.from_structured_output(research_structured)
        else:
            research_decision = ResearchDecision()

        # Build analyst decisions
        analyst_decisions: Dict[str, Any] = {}
        for name, data in [
            ("market", market_structured),
            ("fundamentals", fundamentals_structured),
            ("news", news_structured),
            ("social_media", social_media_structured),
        ]:
            if data:
                analyst_decisions[name] = AnalystDecision.from_structured_output(data).model_dump()

        return cls(
            ticker=context.company_of_interest,
            trade_date=context.trade_date,
            latest_trading_date=context.latest_trading_date,
            analyst_reports=AnalystReports(
                market=context.market_report,
                fundamentals=context.fundamentals_report,
                news=context.news_report,
                social_media=context.sentiment_report,
                market_structured=market_structured,
                fundamentals_structured=fundamentals_structured,
                news_structured=news_structured,
                social_media_structured=social_media_structured,
            ),
            analyst_decisions=analyst_decisions,
            researcher_investment_plan=context.researcher_investment_plan,
            research_decision=research_decision,
            trader_decision=trader_decision,
            portfolio_decision=portfolio_decision,
        )
