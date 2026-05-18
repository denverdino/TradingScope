"""Post-market evaluation module for TradingScope.

This module provides:
- AnalysisRecord: Structured record of trading decisions
- OSSAnalysisStore: OSS-backed discovery and fetching of JSON reports
- AnalysisEvaluator: Scores analysis records against actual market data
  and generates Lessons Learned for agent memory
"""

from .models import AnalysisRecord
from .oss_store import OSSAnalysisStore

__all__ = [
    "AnalysisRecord",
    "OSSAnalysisStore",
]
