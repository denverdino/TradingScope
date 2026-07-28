"""CLI entry point for post-market analysis evaluation.

Usage:
    python -m tradingscope.evaluate                              # evaluate all pending records
    python -m tradingscope.evaluate --tickers MSFT,AAPL,TSLA     # evaluate specific tickers
    python -m tradingscope.evaluate --date 2026-05-01            # evaluate specific date
    python -m tradingscope.evaluate --dry-run                    # preview without side effects
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections import defaultdict
from datetime import datetime

from agentscope import logger

from tradingscope.agents.utils.tracing import setup_tracing, shutdown_tracing


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


async def _run(
    tickers: list[str] | None,
    date: str | None,
    results_dir: str | None,
    dry_run: bool = False,
    email_to: list[str] | None = None,
) -> None:
    from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator
    from tradingscope.agents.utils.context import AgentContext

    context = AgentContext()

    if dry_run:
        logger.info("[DRY RUN] No lessons will be stored and no records will be marked as evaluated")

    evaluator = AnalysisEvaluator(
        model=context.non_thinking_model,
        memory_manager=None,
        results_dir=results_dir,
        dry_run=dry_run,
        middlewares=context.middlewares,
    )
    results = await evaluator.run_batch_evaluation(tickers=tickers, date=date)

    if results:
        logger.info("=== Evaluation Summary ===")
        for i, result in enumerate(results, 1):
            label = f"[{result.ticker}|{result.trade_date}|{result.horizon_days}日|{result.status}]"
            logger.info("Result %d %s:\n  评估: %s\n  教训: %s\n", i, label, result.evaluation, result.lesson)
        logger.info("Total: %d results generated", len(results))

        if email_to:
            _send_evaluation_email(results, date, email_to)
    else:
        logger.info("No pending records to evaluate")


def _send_evaluation_email(results: list, date: str | None, email_to: list[str]) -> None:
    """Build and send evaluation report email grouped by ticker."""
    ticker_results: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        ticker_results[result.ticker][result.trade_date].append(result)

    html_parts = []
    for ticker, date_results in ticker_results.items():
        html_parts.append(f"<h2>{ticker}</h2>")
        for trade_date, items in date_results.items():
            html_parts.append(f"<h3>分析日期：{trade_date}</h3>")
            for item in items:
                html_parts.append(f"<h4>[{item.horizon_days}日|{item.status}]</h4>")
                if item.evaluation:
                    html_parts.append(f"<p>评估： {item.evaluation}</p>")
                if item.lesson:
                    html_parts.append(f"<p>教训： {item.lesson}</p>")
    html_content = "\n".join(html_parts)

    eval_date = date or datetime.now().strftime("%Y-%m-%d")
    subject = f"Stock Evaluation ({eval_date})"
    sender_email = os.getenv("EMAIL_FROM")
    sender_password = os.getenv("EMAIL_PASSWORD")

    if not sender_email or not sender_password:
        logger.error("EMAIL_FROM and EMAIL_PASSWORD must be set to send email.")
        return

    from tradingscope.utils.email_utils import send_html_email

    send_html_email(subject, html_content, email_to, sender_email, sender_password)


def _main() -> None:
    """Parse arguments and run the evaluation CLI."""
    _configure_memory_debug()

    parser = argparse.ArgumentParser(
        description="TradingScope - Post-market analysis evaluation",
    )
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated stock tickers to evaluate (e.g., MSFT,AAPL,TSLA).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Trade date to evaluate (YYYY-MM-DD). Default: previous US trading day.",
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
    parser.add_argument(
        "--email-to",
        default=None,
        help="Comma-separated recipient email addresses for evaluation report.",
    )
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    email_to = [e.strip() for e in args.email_to.split(",") if e.strip()] if args.email_to else None
    asyncio.run(_run(tickers=tickers, date=args.date, results_dir=args.results_dir, dry_run=args.dry_run, email_to=email_to))


def main() -> None:
    """Main entry point for the evaluation CLI."""
    provider = setup_tracing("tradingscope-evaluation")
    try:
        _main()
    finally:
        shutdown_tracing(provider)


if __name__ == "__main__":
    main()
