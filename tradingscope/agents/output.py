"""Strict schema-v2 output contracts for TradingScope agents."""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base class for persisted contracts that reject unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Direction(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Action(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class PositionAdvice(StrEnum):
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class Decision(StrictModel):
    direction: Direction
    action: Action
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(min_length=1)
    reasoning: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_direction_action_conflicts(self) -> Decision:
        if self.action is Action.BUY and self.direction is Direction.BEARISH:
            raise ValueError("buy action cannot use bearish direction")
        if self.action is Action.SELL and self.direction is Direction.BULLISH:
            raise ValueError("sell action cannot use bullish direction")
        return self


class PricePlan(StrictModel):
    entry_price: float | None = Field(default=None, gt=0)
    target_price: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    currency: str = Field(min_length=1)
    invalidation_conditions: list[str] = Field(min_length=1)

    @property
    def risk_reward_ratio(self) -> float | None:
        if self.entry_price is None or self.target_price is None or self.stop_loss is None:
            return None
        risk = self.entry_price - self.stop_loss
        if risk <= 0:
            return None
        return (self.target_price - self.entry_price) / risk


class Evidence(StrictModel):
    claim: str = Field(min_length=1)
    supporting_data: str = Field(min_length=1)
    source: str = Field(min_length=1)
    as_of_date: date | None


class TechnicalIndicator(StrictModel):
    name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    signal: Literal["bullish", "bearish", "neutral"]
    interpretation: str = Field(min_length=1)


class BollingerAnalysis(StrictModel):
    upper_band: float = Field(gt=0)
    middle_band: float = Field(gt=0)
    lower_band: float = Field(gt=0)
    signal: str = Field(min_length=1)


class OptionsAnalysis(StrictModel):
    put_call_ratio: float | None = Field(default=None, ge=0)
    support_levels: list[float]
    resistance_levels: list[float]
    max_pain: float | None = Field(default=None, gt=0)
    interpretation: str = Field(min_length=1)


class FinancialMetric(StrictModel):
    name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    period: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)


class NewsEvent(StrictModel):
    title: str = Field(min_length=1)
    event_date: date | None
    source: str = Field(min_length=1)
    sentiment: Literal["positive", "negative", "neutral"]
    impact: str = Field(min_length=1)


class PlatformSignal(StrictModel):
    platform: str = Field(min_length=1)
    sentiment: Literal["positive", "negative", "neutral"]
    observation: str = Field(min_length=1)


class PriceScenario(StrictModel):
    name: str = Field(min_length=1)
    target_price: float = Field(gt=0)
    timeframe_days: int = Field(ge=1)
    assumptions: list[str]


class AgentOutputBase(StrictModel):
    schema_version: Literal["2.0"]
    agent_name: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    trade_date: date
    latest_trading_date: date
    decision: Decision
    evidence: list[Evidence]
    limitations: list[str]

    @model_validator(mode="after")
    def require_complete_buy_price_plan(self) -> AgentOutputBase:
        price_plan = getattr(self, "price_plan", None)
        if self.decision.action is not Action.BUY:
            return self

        required_fields = ("entry_price", "target_price", "stop_loss")
        missing = [field_name for field_name in required_fields if price_plan is None or getattr(price_plan, field_name) is None]
        if missing:
            raise ValueError(f"buy action requires price_plan fields: {', '.join(missing)}")
        return self


class MarketAnalystOutput(AgentOutputBase):
    market_environment: str = Field(min_length=1)
    price_action: str = Field(min_length=1)
    volume_analysis: str = Field(min_length=1)
    technical_indicators: list[TechnicalIndicator]
    weekly_bollinger: BollingerAnalysis | None
    options_analysis: OptionsAnalysis | None
    signal_strength: Literal["strong", "moderate", "weak"]
    price_plan: PricePlan

    @field_validator("weekly_bollinger", "options_analysis", mode="before")
    @classmethod
    def decode_json_encoded_optional_objects(cls, value: object) -> object:
        """Normalize JSON strings emitted for optional nested tool fields."""
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("must be a valid JSON object or null") from exc


class FundamentalsAnalystOutput(AgentOutputBase):
    company_overview: str = Field(min_length=1)
    financial_performance: list[FinancialMetric]
    valuation_assessment: Literal["overvalued", "fair", "undervalued"]
    earnings_quality: str = Field(min_length=1)
    key_catalysts: list[str]
    key_risks: list[str]
    price_plan: PricePlan


class NewsAnalystOutput(AgentOutputBase):
    overall_sentiment: Literal["positive", "negative", "neutral"]
    key_events: list[NewsEvent]
    macro_environment: str = Field(min_length=1)
    company_specific_impact: str = Field(min_length=1)
    near_term_catalysts: list[str]
    key_risks: list[str]
    price_plan: PricePlan


class SocialMediaAnalystOutput(AgentOutputBase):
    overall_sentiment: Literal["positive", "negative", "neutral"]
    sentiment_score: float = Field(ge=1, le=10)
    platform_signals: list[PlatformSignal]
    key_topics: list[str]
    sentiment_drivers: list[str]
    data_quality: str = Field(min_length=1)
    price_plan: PricePlan


class ResearchManagerOutput(AgentOutputBase):
    bull_viewpoints: list[str]
    bear_viewpoints: list[str]
    adopted_reasoning: list[str]
    strategic_actions: list[str]
    price_scenarios: list[PriceScenario]
    price_plan: PricePlan


class TraderOutput(AgentOutputBase):
    trade_type: Literal["short_term"]
    position_advice: PositionAdvice
    risk_score: float = Field(ge=0, le=1)
    time_stop_days: int | None = Field(default=None, ge=1)
    entry_conditions: list[str]
    execution_steps: list[str]
    risk_factors: list[str]
    price_plan: PricePlan


class PortfolioManagerOutput(AgentOutputBase):
    aggressive_viewpoints: list[str]
    conservative_viewpoints: list[str]
    neutral_viewpoints: list[str]
    adopted_reasoning: list[str]
    position_advice: PositionAdvice
    risk_score: float = Field(ge=0, le=1)
    risk_control_measures: list[str]
    price_plan: PricePlan


class AnalystOutputs(StrictModel):
    market: MarketAnalystOutput
    fundamentals: FundamentalsAnalystOutput
    news: NewsAnalystOutput
    social_media: SocialMediaAnalystOutput


class AnalysisResult(StrictModel):
    schema_version: Literal["2.0"]
    ticker: str = Field(min_length=1)
    trade_date: date
    latest_trading_date: date
    created_at: datetime = Field(default_factory=datetime.now)
    analysts: AnalystOutputs
    research_manager: ResearchManagerOutput
    trader: TraderOutput
    portfolio_manager: PortfolioManagerOutput
