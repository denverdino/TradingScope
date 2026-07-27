"""OSS-backed analysis store for the evaluation module.

Discovers pending evaluations by listing portfolio_manager.json reports
on OSS and tracks evaluated state in a local JSON file.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Set

from tradingscope.agents.output import PortfolioManagerOutput
from tradingscope.default_config import DEFAULT_CONFIG
from tradingscope.utils.oss_structured_output_reader import async_fetch_completed_v2_output

from .models import AnalysisRecord

logger = logging.getLogger(__name__)
ALLOWED_EVALUATION_HORIZONS = (1, 3, 5)


def _validate_horizon_days(horizon_days: int) -> None:
    """Reject evaluation horizons that are not supported by the workflow."""
    if type(horizon_days) is not int or horizon_days not in ALLOWED_EVALUATION_HORIZONS:
        raise ValueError(f"Invalid evaluation horizon {horizon_days}; allowed horizons are 1, 3, 5.")


def build_record_from_portfolio(portfolio: PortfolioManagerOutput) -> AnalysisRecord:
    """Build an evaluation record from a validated portfolio decision."""
    return AnalysisRecord(
        ticker=portfolio.ticker,
        trade_date=portfolio.trade_date.isoformat(),
        direction=portfolio.decision.direction.value,
        action=portfolio.decision.action.value,
        confidence=portfolio.decision.confidence,
        entry_price=portfolio.price_plan.entry_price,
        entry_price_low=portfolio.price_plan.entry_price_low,
        entry_price_high=portfolio.price_plan.entry_price_high,
        target_price=portfolio.price_plan.target_price,
        stop_loss=portfolio.price_plan.stop_loss,
        trade_intent=(portfolio.trade_intent.value if portfolio.trade_intent is not None else None),
        position_advice=portfolio.position_advice.value,
        time_stop_days=portfolio.time_stop_days,
        intent_inferred=portfolio.trade_intent is None,
        reasoning="；".join(portfolio.decision.reasoning)[:100],
        final_decision_summary="；".join(portfolio.adopted_reasoning)[:500],
        status="pending",
    )


class OSSAnalysisStore:
    """Discover and fetch analysis records from OSS JSON reports.

    Reads portfolio_manager.json (structured output) instead of parsing
    markdown reports. Tracks evaluated state in a local JSON file.
    """

    def __init__(self, results_dir: Optional[str] = None) -> None:
        self._results_dir = results_dir or DEFAULT_CONFIG["results_dir"]
        self._tracking_path = os.path.join(self._results_dir, "oss_evaluated.json")

    async def list_pending(
        self,
        before_date: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        date: Optional[str] = None,
    ) -> List[AnalysisRecord]:
        """List pending records from OSS that haven't been evaluated yet.

        Args:
            before_date: Only consider reports with trade_date < this value.
            tickers: List of stock symbols to evaluate (required).
            date: Filter by specific trade date.

        Returns:
            List of AnalysisRecord objects built from OSS JSON reports.
        """
        if not before_date:
            before_date = datetime.now().strftime("%Y-%m-%d")

        if not tickers:
            logger.warning("[OSSStore] --tickers is required for evaluation")
            return []

        candidates: list[tuple[str, str]] = []
        for tkr in tickers:
            if date:
                candidates.append((date, tkr))
            else:
                candidates.extend(self._generate_date_candidates(before_date, tkr))

        pending = [
            (trade_date, ticker)
            for trade_date, ticker in candidates
            if not all(self.is_evaluated(ticker, trade_date, horizon_days) for horizon_days in (1, 3, 5))
        ]

        records: List[AnalysisRecord] = []
        for trade_date, tkr in pending:
            data = await async_fetch_completed_v2_output(trade_date, tkr)
            if not data:
                logger.debug(
                    "[OSSStore] No completed v2 report found for %s/%s",
                    tkr,
                    trade_date,
                )
                continue
            portfolio = PortfolioManagerOutput.model_validate(data)
            record = build_record_from_portfolio(portfolio)
            records.append(record)

        if records:
            logger.info("[OSSStore] Found %d pending reports", len(records))

        return records

    def is_evaluated(self, ticker: str, date: str, horizon_days: int) -> bool:
        """Return whether an evaluation horizon has been completed."""
        _validate_horizon_days(horizon_days)
        evaluated = self._load_evaluated_set()
        horizon_key = f"{date}/{ticker}/{horizon_days}"
        legacy_key = f"{date}/{ticker}"
        return horizon_key in evaluated or (horizon_days == 1 and legacy_key in evaluated)

    def mark_evaluated(self, ticker: str, date: str, horizon_days: int = 1) -> bool:
        """Mark a report as evaluated.

        Args:
            ticker: Stock symbol
            date: Trade date (YYYY-MM-DD)
            horizon_days: Number of post-analysis trading sessions assessed.

        Returns:
            True if successfully updated.
        """
        _validate_horizon_days(horizon_days)
        try:
            evaluated = self._load_evaluated_set()
            evaluated.add(f"{date}/{ticker}/{horizon_days}")
            self._save_evaluated_set(evaluated)
            logger.info(
                "[OSSStore] Marked as evaluated: %s/%s/%s",
                date,
                ticker,
                horizon_days,
            )
            return True
        except Exception as e:
            logger.warning(
                "[OSSStore] Failed to mark evaluated %s/%s/%s: %s",
                date,
                ticker,
                horizon_days,
                e,
            )
            return False

    @staticmethod
    def _generate_date_candidates(before_date: str, ticker: str) -> List[tuple]:
        """Generate the fourteen calendar dates strictly before ``before_date``."""
        cutoff = datetime.strptime(before_date, "%Y-%m-%d")
        return [((cutoff - timedelta(days=offset)).strftime("%Y-%m-%d"), ticker) for offset in range(1, 15)]

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
