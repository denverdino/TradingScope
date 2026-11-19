"""Deterministic post-market evaluation for TradingScope.

- EvaluationRecord: one completed run judged at one horizon
- AnalysisEvaluator: finds pending runs and writes each record once
- build_track_record: statistics over matured evaluations
"""

from .evaluator import AnalysisEvaluator, build_evaluation_record, current_records
from .models import EvaluationRecord
from .oss_store import default_evaluation_store
from .track_record import build_track_record

__all__ = [
    "AnalysisEvaluator",
    "EvaluationRecord",
    "build_evaluation_record",
    "build_track_record",
    "current_records",
    "default_evaluation_store",
]
