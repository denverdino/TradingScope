"""Data models for the evaluation module.

AnalysisRecord stores the key decision basis from each workflow run,
used later by the evaluation process to score against actual market data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structured result of evaluating a single analysis record."""

    ticker: str
    evaluation: str  # 评估结论（方向是否正确、止损情况等）
    lesson: str  # 经验教训
    horizon_days: int = 1
    status: str = "inconclusive"
    entry_triggered: bool = False
    benchmark_return: Optional[float] = None
    strategy_return: Optional[float] = None


@dataclass
class AnalysisRecord:
    """Record of key decision basis from a workflow run.

    Stored as local JSON files. Not subject to the 512-char Memory API limit.
    """

    ticker: str
    trade_date: str  # YYYY-MM-DD
    direction: str  # bullish/bearish/neutral
    action: str  # buy/sell/hold
    confidence: float  # 0-1
    entry_price: Optional[float] = None
    entry_price_low: Optional[float] = None
    entry_price_high: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    trade_intent: Optional[str] = None
    position_advice: Optional[str] = None
    time_stop_days: Optional[int] = None
    intent_inferred: bool = False
    reasoning: str = ""  # core reasoning (~100 chars)
    final_decision_summary: str = ""  # LLM-summarized key decision factors
    status: str = "pending"  # pending/evaluated
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "direction": self.direction,
            "action": self.action,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "entry_price_low": self.entry_price_low,
            "entry_price_high": self.entry_price_high,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "trade_intent": self.trade_intent,
            "position_advice": self.position_advice,
            "time_stop_days": self.time_stop_days,
            "intent_inferred": self.intent_inferred,
            "reasoning": self.reasoning,
            "final_decision_summary": self.final_decision_summary,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisRecord":
        """Deserialize from dictionary."""
        return cls(
            ticker=data["ticker"],
            trade_date=data["trade_date"],
            direction=data["direction"],
            action=data["action"],
            confidence=data["confidence"],
            entry_price=data.get("entry_price"),
            entry_price_low=data.get("entry_price_low"),
            entry_price_high=data.get("entry_price_high"),
            target_price=data.get("target_price"),
            stop_loss=data.get("stop_loss"),
            trade_intent=data.get("trade_intent"),
            position_advice=data.get("position_advice"),
            time_stop_days=data.get("time_stop_days"),
            intent_inferred=data.get("intent_inferred", False),
            reasoning=data.get("reasoning", ""),
            final_decision_summary=data.get("final_decision_summary", ""),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", ""),
        )
