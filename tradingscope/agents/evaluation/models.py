"""Persisted evaluation contracts and the trade-plan input to market simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import AwareDatetime, Field, field_validator, model_validator

from tradingscope.agents.output import Direction, StrictModel, TradeIntent
from tradingscope.utils.oss_structured_output_reader import is_safe_path_component

EVALUATION_SCHEMA_VERSION = "1.0"


@dataclass
class AnalysisRecord:
    """Decision and price plan of one stage output, as simulated by market_outcome."""

    analysis_date: str
    direction: str
    trade_intent: str
    execution_price: float | None = None
    execution_price_low: float | None = None
    execution_price_high: float | None = None
    target_price: float | None = None
    stop_loss: float | None = None
    position_advice: str | None = None
    time_stop_days: int | None = None


class ShockClass(StrEnum):
    SCHEDULED = "scheduled"
    UNSCHEDULED = "unscheduled"
    UNKNOWN = "unknown"


class SampleClass(StrEnum):
    CLEAN = "clean"
    SHOCK_EXCLUDED = "shock_excluded"


class ShockSession(StrictModel):
    session_date: date
    excess_return: float
    excess_gap: float
    classification: ShockClass


class MarketOutcome(StrictModel):
    reference_session: date
    reference_close: float = Field(gt=0)
    daily_atr_pct: float = Field(ge=0)
    atr_threshold: float = Field(ge=0)
    underlying_return: float
    benchmark_symbol: str = Field(min_length=1)
    benchmark_return: float
    excess_return: float
    shock_sessions: list[ShockSession]
    sample_class: SampleClass


OutcomeStatus = Literal["correct", "incorrect", "inconclusive"]


class TradeOutcome(StrictModel):
    trade_intent: TradeIntent
    simulated: bool
    status: Literal["correct", "incorrect", "inconclusive", "not_filled"]
    filled: bool
    fill_price: float | None
    exit_reason: str | None
    strategy_return: float | None
    strategy_return_basis: str | None
    residual_exit_reason: str | None
    residual_exit_date: str | None
    residual_exit_price: float | None
    limitations: list[str]


def effective_return_basis(trade: TradeOutcome) -> str:
    return trade.strategy_return_basis or (
        "position_return" if trade.trade_intent in {TradeIntent.OPEN_LONG, TradeIntent.OPEN_SHORT} else "exit_relative_performance"
    )


class StageEvaluation(StrictModel):
    stage: str = Field(min_length=1)
    direction: Direction
    confidence: float = Field(ge=0, le=1)
    direction_status: OutcomeStatus
    excess_direction_status: OutcomeStatus
    trade: TradeOutcome | None = None


class EvaluationRecord(StrictModel):
    """One completed run judged at one horizon; written once and never modified."""

    schema_version: Literal["1.0"] = EVALUATION_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    code_version: str | None
    models: dict[str, str]
    ticker: str = Field(min_length=1)
    analysis_date: date
    horizon: int = Field(ge=1)
    horizon_end_session: date
    evaluated_at: AwareDatetime
    forecast_started_at: AwareDatetime
    forecast_completed_at: AwareDatetime
    market: MarketOutcome
    stages: list[StageEvaluation] = Field(min_length=1)
    source_outputs: dict[str, dict[str, Any]]

    @field_validator("ticker", "run_id")
    @classmethod
    def safe_path_component(cls, value: str) -> str:
        if not is_safe_path_component(value):
            raise ValueError("identity must be a single safe path component")
        return value

    @model_validator(mode="after")
    def consistent_sessions(self) -> EvaluationRecord:
        if self.horizon_end_session < self.analysis_date:
            raise ValueError("horizon must end on or after the analysis date")
        if self.market.reference_session >= self.analysis_date:
            raise ValueError("reference session must precede the analysis date")
        names = [stage.stage for stage in self.stages]
        if len(set(names)) != len(names) or set(names) != set(self.source_outputs):
            raise ValueError("stage results must uniquely match source outputs")
        return self

    def stage(self, name: str) -> StageEvaluation | None:
        return next((stage for stage in self.stages if stage.stage == name), None)

    def evaluation_id(self, stage: str) -> str:
        return f"{self.analysis_date.isoformat()}/{self.ticker}/{self.run_id}/h{self.horizon}/{stage}"
