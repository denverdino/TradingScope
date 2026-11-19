"""Strict schema-v3 output contracts for TradingScope agents."""

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


class TradeIntent(StrEnum):
    OPEN_LONG = "open_long"
    REDUCE_LONG = "reduce_long"
    CLOSE_LONG = "close_long"
    OPEN_SHORT = "open_short"
    COVER_SHORT = "cover_short"
    HOLD = "hold"


class Decision(StrictModel):
    direction: Direction
    action: Action
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(min_length=1)
    reasoning: list[str] = Field(min_length=1)


class PricePlan(StrictModel):
    execution_price: float | None = Field(
        default=None,
        gt=0,
        description="代表执行价。给出执行区间时必须满足 execution_price_low <= execution_price <= execution_price_high；不得把当前价填作代表价，除非它位于区间内。",
    )
    execution_price_low: float | None = Field(
        default=None,
        gt=0,
        description="执行区间下沿。与 execution_price_high 必须同时给出或同时留空；素材只有单一价位时两者都留空。",
    )
    execution_price_high: float | None = Field(
        default=None,
        gt=0,
        description="执行区间上沿，必须不低于 execution_price_low。与 execution_price_low 必须同时给出或同时留空。",
    )
    target_price: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    currency: str = Field(min_length=1)
    invalidation_conditions: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_execution_range(self) -> PricePlan:
        if (self.execution_price_low is None) != (self.execution_price_high is None):
            raise ValueError("execution_price_low and execution_price_high must be provided together")
        if self.execution_price_low is None or self.execution_price_high is None:
            return self
        if self.execution_price_low > self.execution_price_high:
            raise ValueError("execution price range must be ordered from low to high")
        if self.execution_price is not None and not self.execution_price_low <= self.execution_price <= self.execution_price_high:
            raise ValueError("execution_price must be within the execution price range")
        return self

    @property
    def risk_reward_ratio(self) -> float | None:
        return self.risk_reward_ratio_for(Direction.BULLISH)

    def risk_reward_ratio_for(self, direction: Direction) -> float | None:
        if self.execution_price is None or self.target_price is None or self.stop_loss is None:
            return None
        if direction is Direction.BULLISH:
            risk = self.execution_price - self.stop_loss
            reward = self.target_price - self.execution_price
        elif direction is Direction.BEARISH:
            risk = self.stop_loss - self.execution_price
            reward = self.execution_price - self.target_price
        else:
            return None
        if risk <= 0 or reward <= 0:
            return None
        return reward / risk


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
    put_call_ratio: float | None = Field(default=None, ge=0, description="仅使用素材中有依据的 PCR，并在 interpretation 中标明 OI/成交量口径；不可用时 null。")
    support_levels: list[float] = Field(description="仅填素材采用的期权 OI 支撑参考位；不可用或被弃用时空列表 []，不得填入技术面价位。")
    resistance_levels: list[float] = Field(description="仅填素材采用的期权 OI 阻力参考位；不可用或被弃用时空列表 []，不得填入技术面价位。")
    max_pain: float | None = Field(default=None, gt=0, description="仅填有效且唯一的 Max Pain；缺失、非唯一或被弃用时 null，区间只在 interpretation 说明。")
    interpretation: str = Field(
        min_length=1, description="保留选用到期日、OI 更新时间与覆盖完整性限制、PCR 口径和价位弃用原因；缺数与真实零分开，不将技术面价位当期权确认。"
    )


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
    timeframe_days: int = Field(ge=1, description="情景的交易日期限。新生成情景必须为 1，只预测下一交易日截至收盘的价格。")
    assumptions: list[str]


class AgentOutputBase(StrictModel):
    schema_version: Literal["3.0"]
    agent_name: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    analysis_date: date
    last_completed_session: date
    decision: Decision
    evidence: list[Evidence]
    limitations: list[str]

    @model_validator(mode="after")
    def reject_direction_action_conflicts(self) -> AgentOutputBase:
        if self.decision.action is Action.BUY and self.decision.direction is Direction.BEARISH:
            raise ValueError("buy action cannot use bearish direction")
        is_reduction = isinstance(self, (TraderOutput, PortfolioManagerOutput)) and self.trade_intent is TradeIntent.REDUCE_LONG
        if self.decision.action is Action.SELL and self.decision.direction is Direction.BULLISH and not is_reduction:
            raise ValueError("sell action cannot use bullish direction")
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
    data_quality: str = Field(
        min_length=1,
        description="说明实际数据来源与质量；独立社交样本缺失只在本字段说明一次，新闻情绪代理身份与代表性边界一并放在此处，不再复制到 limitations。",
    )
    limitations: list[str] = Field(
        description="不重复 data_quality 中的独立社交样本缺失或新闻情绪代理身份说明；仅保留其他限制，如数据滞后、事件结果未核验、缺少可比时间序列。若素材将采样缺失与其他限制合并表述，仅将采样缺失部分归入 data_quality，其余限制保留；没有其他限制时为空列表 []。",
    )
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
    trade_intent: TradeIntent | None = None
    time_stop_days: int | None = Field(
        default=None,
        ge=1,
        description="从分析日期对应首个交易日起计数的独立持仓管理期限，可大于 1；不延长下一交易日预测的有效期。纯退出或无仓位时留空。",
    )
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
    trade_intent: TradeIntent | None = None
    time_stop_days: int | None = Field(
        default=None,
        ge=1,
        description="从分析日期对应首个交易日起计数的独立持仓管理期限，可大于 1；不延长下一交易日预测的有效期。纯退出或无仓位时留空。",
    )
    risk_control_measures: list[str]
    price_plan: PricePlan


class AnalystOutputs(StrictModel):
    market: MarketAnalystOutput
    fundamentals: FundamentalsAnalystOutput
    news: NewsAnalystOutput
    social_media: SocialMediaAnalystOutput


class AnalysisResult(StrictModel):
    schema_version: Literal["3.0"]
    ticker: str = Field(min_length=1)
    analysis_date: date
    last_completed_session: date
    created_at: datetime = Field(default_factory=datetime.now)
    analysts: AnalystOutputs
    research_manager: ResearchManagerOutput
    trader: TraderOutput
    portfolio_manager: PortfolioManagerOutput
