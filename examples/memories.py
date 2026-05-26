#!/usr/bin/env python3
"""Example script to review saved analysis records.

The long-term memory system (Model Studio Memory API) has been removed
in the AgentScope 2.0 migration. This script retains the analysis record
review functionality which reads from local JSON files.

Usage:
    # Review saved analysis records for a stock
    python -m examples.memories --review AAPL

    # Review analysis records for a stock on a specific date
    python -m examples.memories --review AAPL --date 2026-05-01

    # Review all saved analysis records
    python -m examples.memories --review all
"""

import argparse
import json
import os

from tradingscope.default_config import DEFAULT_CONFIG


def review_records(ticker: str | None, date: str | None, results_dir: str | None) -> None:
    """Review saved analysis records for a stock.

    Args:
        ticker: Stock symbol to filter, or None to show all
        date: Specific date to filter (YYYY-MM-DD), or None for all dates
        results_dir: Path to results directory
    """
    base_dir = os.path.join(results_dir or DEFAULT_CONFIG["results_dir"], "analysis_records")

    if not os.path.isdir(base_dir):
        print(f"\n  No analysis records found (directory not found: {base_dir})")
        return

    records = []
    for date_dir in sorted(os.listdir(base_dir)):
        if date and date_dir != date:
            continue
        date_path = os.path.join(base_dir, date_dir)
        if not os.path.isdir(date_path):
            continue
        for filename in sorted(os.listdir(date_path)):
            if not filename.endswith(".json"):
                continue
            file_ticker = filename[:-5]  # remove .json
            if ticker and file_ticker.upper() != ticker.upper():
                continue
            filepath = os.path.join(date_path, filename)
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                records.append(data)
            except Exception as e:
                print(f"  Warning: Failed to load {filepath}: {e}")

    if not records:
        filter_desc = []
        if ticker:
            filter_desc.append(f"ticker={ticker}")
        if date:
            filter_desc.append(f"date={date}")
        filter_str = ", ".join(filter_desc) if filter_desc else "no filter"
        print(f"\n  No analysis records found ({filter_str})")
        return

    print(f"\n{'=' * 70}")
    print(f"  Analysis Records ({len(records)} found)")
    print(f"{'=' * 70}")

    for i, rec in enumerate(records):
        if i > 0:
            print(f"\n{'-' * 70}")
        print(f"\n  [{rec.get('trade_date', '?')}] {rec.get('ticker', '?')}")
        print(f"  Direction:   {rec.get('direction', '?')}")
        print(f"  Action:      {rec.get('action', '?')}")
        print(f"  Confidence:  {rec.get('confidence', '?')}")
        if rec.get("entry_price") is not None:
            print(f"  Entry Price: {rec['entry_price']}")
        if rec.get("target_price") is not None:
            print(f"  Target:      {rec['target_price']}")
        if rec.get("stop_loss") is not None:
            print(f"  Stop Loss:   {rec['stop_loss']}")
        if rec.get("reasoning"):
            print(f"  Reasoning:   {rec['reasoning']}")
        print(f"  Status:      {rec.get('status', '?')}")
        print(f"  Created:     {rec.get('created_at', '?')}")
        if rec.get("final_decision_summary"):
            print("\n  --- Decision Summary ---")
            print(f"  {rec['final_decision_summary']}")

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Review saved analysis records for TradingScope")
    parser.add_argument(
        "--review",
        nargs="?",
        const="all",
        default="all",
        metavar="TICKER",
        help="Review saved analysis records. Use 'all' or omit ticker to show all.",
    )
    parser.add_argument("--date", help="Filter analysis records by date (YYYY-MM-DD)")
    parser.add_argument("--results-dir", help="Path to results directory")

    args = parser.parse_args()

    ticker = None if args.review.lower() == "all" else args.review
    review_records(ticker=ticker, date=args.date, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
