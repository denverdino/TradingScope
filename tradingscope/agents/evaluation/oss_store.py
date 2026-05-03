"""OSS-backed analysis store for the evaluation module.

Discovers pending evaluations by listing portfolio_manager.md reports
on OSS and tracks evaluated state in a local JSON file.

When OSS listing is unavailable (e.g., bucket policy restricts it),
falls back to probing recent dates directly via GetObject.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Set

from tradingscope.default_config import DEFAULT_CONFIG
from tradingscope.utils.oss_report_reader import (
    async_fetch_report,
)

from .models import AnalysisRecord
from .report_parser import build_analysis_record

logger = logging.getLogger(__name__)


class OSSAnalysisStore:
    """Discover and fetch analysis reports from OSS.

    Uses OSS listing to find portfolio_manager.md reports, and tracks
    which reports have been evaluated in a local JSON file.

    When listing is not available (permissions, bucket policy), falls
    back to probing specific dates via GetObject.
    """

    def __init__(self, results_dir: Optional[str] = None) -> None:
        self._results_dir = results_dir or DEFAULT_CONFIG["results_dir"]
        self._tracking_path = os.path.join(self._results_dir, "oss_evaluated.json")

    async def list_pending(
        self,
        before_date: Optional[str] = None,
        ticker: Optional[str] = None,
        date: Optional[str] = None,
    ) -> List[AnalysisRecord]:
        """List pending reports from OSS that haven't been evaluated yet.

        Strategy:
        1. If date+ticker are specified, directly fetch that report.
        2. If only ticker is specified, probe recent dates via GetObject.
        3. Otherwise, try OSS listing first; if it fails, probe recent dates.

        Args:
            before_date: Only consider reports with trade_date < this value.
            ticker: Filter by stock symbol.
            date: Filter by specific trade date.

        Returns:
            List of AnalysisRecord objects parsed from OSS reports.
        """
        if not before_date:
            before_date = datetime.now().strftime("%Y-%m-%d")

        # Determine which (date, ticker) pairs to check
        if not ticker:
            logger.warning("[OSSStore] --ticker is required for evaluation")
            return []

        if date:
            candidates = [(date, ticker)]
        else:
            # Default: evaluate yesterday's report only
            candidates = self._generate_date_candidates(before_date, ticker)

        # Filter out already-evaluated
        evaluated = self._load_evaluated_set()
        pending = [(d, t) for d, t in candidates if f"{d}/{t}" not in evaluated]

        # Fetch and parse each pending report
        records: List[AnalysisRecord] = []
        for trade_date, tkr in pending:
            report_text = await async_fetch_report(trade_date, tkr)
            if not report_text:
                continue

            record = build_analysis_record(tkr, trade_date, report_text)
            records.append(record)

        if records:
            logger.info("[OSSStore] Found %d pending reports", len(records))

        return records

    def mark_evaluated(self, ticker: str, date: str) -> bool:
        """Mark a report as evaluated.

        Args:
            ticker: Stock symbol
            date: Trade date (YYYY-MM-DD)

        Returns:
            True if successfully updated.
        """
        try:
            evaluated = self._load_evaluated_set()
            evaluated.add(f"{date}/{ticker}")
            self._save_evaluated_set(evaluated)
            logger.info("[OSSStore] Marked as evaluated: %s/%s", date, ticker)
            return True
        except Exception as e:
            logger.warning("[OSSStore] Failed to mark evaluated %s/%s: %s", date, ticker, e)
            return False

    @staticmethod
    def _generate_date_candidates(before_date: str, ticker: str) -> List[tuple]:
        """Generate (date, ticker) pair for yesterday only."""
        yesterday = (datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        return [(yesterday, ticker)]

    def _load_evaluated_set(self) -> Set[str]:
        """Load the set of evaluated report keys from local tracking file."""
        if not os.path.exists(self._tracking_path):
            return set()
        try:
            with open(self._tracking_path, encoding="utf-8") as f:
                data = json.load(f)
            return set(data) if isinstance(data, list) else set()
        except Exception:
            return set()

    def _save_evaluated_set(self, evaluated: Set[str]) -> None:
        """Persist the evaluated set to the local tracking file."""
        os.makedirs(os.path.dirname(self._tracking_path), exist_ok=True)
        with open(self._tracking_path, "w", encoding="utf-8") as f:
            json.dump(sorted(evaluated), f, ensure_ascii=False, indent=2)
