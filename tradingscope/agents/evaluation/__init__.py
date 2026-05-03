"""Post-market evaluation module for TradingScope.

This module provides:
- AnalysisRecord: Structured record of trading decisions
- OSSAnalysisStore: OSS-backed discovery and fetching of reports
- AnalysisEvaluator: Scores analysis records against actual market data
  and generates Lessons Learned for agent memory
"""

from .models import AnalysisRecord
from .oss_store import OSSAnalysisStore
from .report_parser import parse_prediction_from_report

__all__ = [
    "AnalysisRecord",
    "OSSAnalysisStore",
    "parse_prediction_from_report",
]
