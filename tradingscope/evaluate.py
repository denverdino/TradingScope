"""CLI entry point for deterministic post-market evaluation.

Usage:
    python -m tradingscope.evaluate --tickers MSFT,AAPL              # pending runs from the previous 14 calendar dates
    python -m tradingscope.evaluate --tickers AAPL --date 2026-05-01 # one analysis date
    python -m tradingscope.evaluate --tickers AAPL --dry-run         # compute and print only
"""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from functools import partial

from agentscope import logger

from tradingscope.agents.utils.tracing import setup_tracing, shutdown_tracing
from tradingscope.default_config import DEFAULT_CONFIG


def _run(
    tickers: list[str],
    date: str | None,
    results_dir: str | None,
    dry_run: bool = False,
    email_to: list[str] | None = None,
) -> None:
    from tradingscope.agents.evaluation.evaluator import AnalysisEvaluator, current_records
    from tradingscope.agents.evaluation.oss_store import default_evaluation_store
    from tradingscope.agents.evaluation.report import record_headline, render_track_record_text
    from tradingscope.agents.evaluation.track_record import build_track_record

    if dry_run:
        logger.info("[DRY RUN] Evaluation records are not written and no email is sent")

    store = default_evaluation_store(results_dir)
    records = AnalysisEvaluator(store, results_dir=results_dir, dry_run=dry_run).run(tickers, analysis_date=date)
    if not records:
        logger.info("No mature evaluation results available")
        return

    for record in records:
        logger.info(record_headline(record))

    from tradingscope.utils.oss_structured_output_reader import fetch_completed_manifest

    history = current_records(
        store.list_records(tickers) + (records if dry_run else []),
        fetch_manifest=partial(fetch_completed_manifest, results_dir=results_dir),
    )
    config = DEFAULT_CONFIG["evaluation"]
    start_date = config["track_record_start_date"]
    track_record = build_track_record(
        history,
        min_samples=config["track_record_min_samples"],
        confidence_buckets=config["confidence_buckets"],
        start_date=_parse_date(start_date) if start_date else None,
    )
    logger.info(render_track_record_text(track_record))
    logger.info("Total: %d evaluation records", len(records))

    if email_to and not dry_run:
        _send_evaluation_email(records, track_record, date, email_to)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _send_evaluation_email(records: list, track_record, date: str | None, email_to: list[str]) -> None:
    from tradingscope.agents.evaluation.report import render_records_html, render_track_record_html

    html_content = f"{render_records_html(records)}\n{render_track_record_html(track_record)}"
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
        help="Analysis date to evaluate (YYYY-MM-DD). Default: pending runs from the previous 14 calendar dates.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for local evaluation records when OSS is not configured. Default: from config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute and print evaluations without writing records or sending email.",
    )
    parser.add_argument(
        "--email-to",
        default=None,
        help="Comma-separated recipient email addresses for evaluation report.",
    )
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    email_to = [e.strip() for e in args.email_to.split(",") if e.strip()] if args.email_to else None
    _run(tickers=tickers, date=args.date, results_dir=args.results_dir, dry_run=args.dry_run, email_to=email_to)


def main() -> None:
    """Main entry point for the evaluation CLI."""
    provider = setup_tracing("tradingscope-evaluation")
    try:
        _main()
    finally:
        shutdown_tracing(provider)


if __name__ == "__main__":
    main()
