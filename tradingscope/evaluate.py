"""CLI entry point for post-market analysis evaluation.

Usage:
    python -m tradingscope.evaluate                    # evaluate all pending records
    python -m tradingscope.evaluate --ticker AAPL      # evaluate specific ticker
    python -m tradingscope.evaluate --date 2026-05-01  # evaluate specific date
    python -m tradingscope.evaluate --dry-run          # preview without side effects
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timedelta

from agentscope import logger


def _configure_memory_debug() -> None:
    """Enable DEBUG logging for memory API when MEMORY_DEBUG env var is set."""
    if not os.getenv("MEMORY_DEBUG"):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"),
    )

    for name in (
        "agentscope_runtime.tools.modelstudio_memory",
        "tradingscope.agents.utils.memory",
        "tradingscope.agents.evaluation.evaluator",
    ):
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        log.addHandler(handler)


async def _run(ticker: str | None, date: str | None, results_dir: str | None, dry_run: bool = False) -> None:
    from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator

    if dry_run:
        logger.info("[DRY RUN] No lessons will be stored and no records will be marked as evaluated")
        evaluator = AnalysisEvaluator(
            memory_manager=None,
            results_dir=results_dir,
            dry_run=True,
        )
        lessons = await evaluator.run_batch_evaluation(ticker=ticker, date=date)
    else:
        from tradingscope.agents.utils.memory_manager import FinancialMemoryManager

        async with FinancialMemoryManager() as memory_manager:
            evaluator = AnalysisEvaluator(
                memory_manager=memory_manager,
                results_dir=results_dir,
            )
            lessons = await evaluator.run_batch_evaluation(ticker=ticker, date=date)

    if lessons:
        logger.info("=== Evaluation Summary ===")
        for i, lesson in enumerate(lessons, 1):
            logger.info("Lesson %d:\n%s\n", i, lesson)
        logger.info("Total: %d lessons generated", len(lessons))
    else:
        logger.info("No pending records to evaluate")


def main() -> None:
    """Main entry point for the evaluation CLI."""
    _configure_memory_debug()

    parser = argparse.ArgumentParser(
        description="TradingScope - Post-market analysis evaluation",
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Stock ticker to evaluate (e.g., AAPL).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Trade date to evaluate (YYYY-MM-DD). Default: yesterday.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory containing analysis_records/. Default: from config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview evaluation without storing lessons or marking records.",
    )
    args = parser.parse_args()

    asyncio.run(_run(ticker=args.ticker, date=args.date, results_dir=args.results_dir, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
