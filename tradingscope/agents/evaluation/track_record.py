"""Deterministic Track Record statistics over matured evaluations."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Iterable, Sequence

from pydantic import Field

from tradingscope.agents.output import StrictModel

from .market_outcome import EVALUATION_CONFIG, AssessmentStatus
from .models import EvaluationRecord, SampleClass, StageEvaluation, effective_return_basis

ALL_TICKERS = "ALL"
ANALYST_STAGES = ("market_analyst", "fundamentals_analyst", "news_analyst", "social_media_analyst")
TRADE_STAGES = ("trader", "portfolio_manager")


class DirectionSkill(StrictModel):
    """Excess-return direction calls on clean evaluations."""

    n: int = 0
    correct: int = 0
    incorrect: int = 0
    inconclusive: int = 0
    hit_rate: float | None = None
    inconclusive_share: float | None = None


class CalibrationBucket(StrictModel):
    lower: float
    upper: float | None
    skill: DirectionSkill


class ReturnPerformance(StrictModel):
    trade_intent: str
    return_basis: str
    n: int
    mean_strategy_return: float | None
    mean_strategy_minus_underlying: float | None


class RiskControl(StrictModel):
    """Simulated trades across all evaluations, including shock sessions."""

    n: int = 0
    filled: int = 0
    stop_losses: int = 0
    fill_rate: float | None = None
    stop_rate: float | None = None
    mean_strategy_return: float | None = None
    mean_strategy_minus_underlying: float | None = None
    residual_stop_losses: int = 0
    residual_stop_rate: float | None = None
    return_performance: list[ReturnPerformance] = Field(default_factory=list)


class TrackRecordCell(StrictModel):
    scope: str
    stage: str
    horizon: int
    direction: DirectionSkill
    calibration: list[CalibrationBucket]
    risk: RiskControl | None = None


class AnalystAttribution(StrictModel):
    """Portfolio manager direction skill when it agreed or disagreed with one analyst."""

    scope: str
    horizon: int
    analyst_stage: str
    agreed: DirectionSkill
    disagreed: DirectionSkill


class TrackRecord(StrictModel):
    covered_through: date | None
    start_date: date | None
    min_samples: int = Field(ge=1)
    cells: list[TrackRecordCell]
    attribution: list[AnalystAttribution]


class _SkillCounter:
    def __init__(self) -> None:
        self.counts = dict.fromkeys((AssessmentStatus.CORRECT, AssessmentStatus.INCORRECT, AssessmentStatus.INCONCLUSIVE), 0)

    def add(self, status: str) -> None:
        self.counts[AssessmentStatus(status)] += 1

    def result(self, min_samples: int) -> DirectionSkill:
        correct = self.counts[AssessmentStatus.CORRECT]
        incorrect = self.counts[AssessmentStatus.INCORRECT]
        inconclusive = self.counts[AssessmentStatus.INCONCLUSIVE]
        n = correct + incorrect + inconclusive
        enough = n >= min_samples
        decided = correct + incorrect
        return DirectionSkill(
            n=n,
            correct=correct,
            incorrect=incorrect,
            inconclusive=inconclusive,
            hit_rate=correct / decided if enough and decided else None,
            inconclusive_share=inconclusive / n if enough else None,
        )


class _RiskCounter:
    def __init__(self) -> None:
        self.n = 0
        self.filled = 0
        self.stop_losses = 0
        self.residual_stop_losses = 0
        self.residual_positions = 0
        self.returns: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)

    def add(self, stage: StageEvaluation, underlying_return: float) -> None:
        trade = stage.trade
        if trade is None or not trade.simulated:
            return
        self.n += 1
        if not trade.filled:
            return
        self.filled += 1
        if trade.exit_reason == "stop_loss":
            self.stop_losses += 1
        if trade.residual_exit_reason is not None:
            self.residual_positions += 1
            self.residual_stop_losses += trade.residual_exit_reason == "stop_loss"
        if trade.strategy_return is not None:
            key = (str(trade.trade_intent), effective_return_basis(trade))
            self.returns[key].append((trade.strategy_return, trade.strategy_return - underlying_return))

    def result(self, min_samples: int) -> RiskControl:
        enough = self.n >= min_samples
        performance = [
            ReturnPerformance(
                trade_intent=intent,
                return_basis=basis,
                n=len(returns),
                mean_strategy_return=sum(sample[0] for sample in returns) / len(returns) if len(returns) >= min_samples else None,
                mean_strategy_minus_underlying=sum(sample[1] for sample in returns) / len(returns) if len(returns) >= min_samples else None,
            )
            for (intent, basis), returns in sorted(self.returns.items())
        ]
        sole = performance[0] if len(performance) == 1 else None
        return RiskControl(
            n=self.n,
            filled=self.filled,
            stop_losses=self.stop_losses,
            fill_rate=self.filled / self.n if enough else None,
            stop_rate=self.stop_losses / self.filled if enough and self.filled else None,
            mean_strategy_return=sole.mean_strategy_return if sole else None,
            mean_strategy_minus_underlying=sole.mean_strategy_minus_underlying if sole else None,
            residual_stop_losses=self.residual_stop_losses,
            residual_stop_rate=self.residual_stop_losses / self.residual_positions if self.residual_positions >= min_samples else None,
            return_performance=performance,
        )


def eligible_records(
    records: Iterable[EvaluationRecord],
    *,
    covered_through: date | None,
    start_date: date | None,
) -> list[EvaluationRecord]:
    """Keep evaluations matured by ``covered_through`` and on or after ``start_date``."""
    return [
        record
        for record in records
        if record.horizon in EVALUATION_CONFIG["horizons"]
        and (covered_through is None or record.horizon_end_session <= covered_through)
        and (start_date is None or record.analysis_date >= start_date)
    ]


def build_track_record(
    records: Iterable[EvaluationRecord],
    *,
    min_samples: int,
    confidence_buckets: Sequence[float],
    covered_through: date | None = None,
    start_date: date | None = None,
) -> TrackRecord:
    edges = [0.0, *confidence_buckets]
    skills: dict[tuple[str, str, int], _SkillCounter] = defaultdict(_SkillCounter)
    buckets: dict[tuple[str, str, int, int], _SkillCounter] = defaultdict(_SkillCounter)
    risks: dict[tuple[str, str, int], _RiskCounter] = defaultdict(_RiskCounter)
    attribution: dict[tuple[str, int, str, bool], _SkillCounter] = defaultdict(_SkillCounter)
    keys: set[tuple[str, str, int]] = set()

    for record in eligible_records(records, covered_through=covered_through, start_date=start_date):
        clean = record.market.sample_class is SampleClass.CLEAN
        portfolio = record.stage("portfolio_manager")
        for scope in (record.ticker, ALL_TICKERS):
            for stage in record.stages:
                key = (scope, stage.stage, record.horizon)
                keys.add(key)
                if clean:
                    skills[key].add(stage.excess_direction_status)
                    bucket = max(index for index, edge in enumerate(edges) if stage.confidence >= edge)
                    buckets[(*key, bucket)].add(stage.excess_direction_status)
                if stage.stage in TRADE_STAGES:
                    risks[key].add(stage, record.market.underlying_return)
            if clean and portfolio is not None:
                for analyst in ANALYST_STAGES:
                    analyst_stage = record.stage(analyst)
                    if analyst_stage is not None:
                        agreed = analyst_stage.direction == portfolio.direction
                        attribution[(scope, record.horizon, analyst, agreed)].add(portfolio.excess_direction_status)

    cells = []
    for scope, stage, horizon in sorted(keys, key=lambda key: (key[0] != ALL_TICKERS, key)):
        calibration = [
            CalibrationBucket(
                lower=edges[index],
                upper=edges[index + 1] if index + 1 < len(edges) else None,
                skill=buckets[(scope, stage, horizon, index)].result(min_samples),
            )
            for index in range(len(edges))
        ]
        cells.append(
            TrackRecordCell(
                scope=scope,
                stage=stage,
                horizon=horizon,
                direction=skills[(scope, stage, horizon)].result(min_samples),
                calibration=calibration,
                risk=risks[(scope, stage, horizon)].result(min_samples) if stage in TRADE_STAGES else None,
            )
        )

    attributions = [
        AnalystAttribution(
            scope=scope,
            horizon=horizon,
            analyst_stage=analyst,
            agreed=attribution[(scope, horizon, analyst, True)].result(min_samples),
            disagreed=attribution[(scope, horizon, analyst, False)].result(min_samples),
        )
        for scope, horizon, analyst in sorted({key[:3] for key in attribution}, key=lambda key: (key[0] != ALL_TICKERS, key))
    ]
    return TrackRecord(
        covered_through=covered_through,
        start_date=start_date,
        min_samples=min_samples,
        cells=cells,
        attribution=attributions,
    )
