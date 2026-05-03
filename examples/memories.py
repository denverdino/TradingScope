#!/usr/bin/env python3
"""Example script to view and manage long-term memories.

Usage:
    # View lessons learned memories
    python -m examples.memories

    # Search memories with a specific query
    python -m examples.memories --query "技术面强但基本面疲软"

    # Add a trading lesson to memory
    python -m examples.memories --add \
        --situation "RSI超卖，MACD金叉" \
        --decision "买入" \
        --outcome "盈利8%" \
        --lesson "RSI超卖配合MACD金叉是较强的买入信号"

    # Clear all lessons learned memories (requires confirmation)
    python -m examples.memories --clear

    # Force clear without confirmation (use with caution!)
    python -m examples.memories --clear --force

    # Manage a specific memory namespace by name
    python -m examples.memories --namespace trader --clear

    # Review saved analysis records for a stock
    python -m examples.memories --review AAPL

    # Review analysis records for a stock on a specific date
    python -m examples.memories --review AAPL --date 2026-05-01

    # Review all saved analysis records
    python -m examples.memories --review all
"""

import argparse
import asyncio
import json
import os

from tradingscope.agents.utils.memory import ModelStudioLongTermMemory
from tradingscope.default_config import DEFAULT_CONFIG

try:
    from agentscope_runtime.tools.modelstudio_memory import (
        DeleteMemory,
        DeleteMemoryInput,
        ListMemory,
        ListMemoryInput,
    )

    MEMORY_API_AVAILABLE = True
except ImportError:
    MEMORY_API_AVAILABLE = False

# Default namespace after refactoring
DEFAULT_NAMESPACE = "lessons_learned"


async def view_memories(namespace: str, query: str, top_k: int, user_name: str) -> None:
    """Search and display memories for a namespace."""
    memory = ModelStudioLongTermMemory(
        agent_name=namespace,
        user_name=user_name,
        top_k=top_k,
    )

    try:
        print(f"\n{'=' * 60}")
        print(f"  Namespace: {namespace}")
        print(f"  User ID:   {memory.user_id}")
        print(f"  Query:     {query}")
        print(f"  Top K:     {top_k}")
        print(f"{'=' * 60}")

        result = await memory.retrieve_from_memory(query)
        if result == "Memory system unavailable":
            print("\n  [ERROR] Memory API not available. Check DASHSCOPE_API_KEY.")
        elif result == "No relevant memories found":
            print("\n  No memories found.")
        else:
            print(f"\n{result}")

    finally:
        await memory.close()


async def add_lesson(
    namespace: str,
    situation: str,
    decision: str,
    outcome: str,
    lesson: str,
    user_name: str,
) -> None:
    """Add a trading lesson to memory."""
    memory = ModelStudioLongTermMemory(
        agent_name=namespace,
        user_name=user_name,
    )

    try:
        print(f"\nAdding trading lesson to '{namespace}' memory...")
        success = await memory.add_trading_lesson(
            situation=situation,
            decision=decision,
            outcome=outcome,
            lesson=lesson,
        )
        if success:
            print("  Trading lesson added successfully!")
        else:
            print("  Failed to add trading lesson. Check DASHSCOPE_API_KEY.")
    finally:
        await memory.close()


async def clear_memories(namespace: str, user_name: str, force: bool = False) -> bool:
    """Clear all memories for a namespace.

    Args:
        namespace: Memory namespace name
        user_name: User name prefix for memory namespace
        force: Skip confirmation if True

    Returns:
        True if cleared successfully, False otherwise
    """
    if not MEMORY_API_AVAILABLE:
        print("  [ERROR] Memory API not available. Check DASHSCOPE_API_KEY.")
        return False

    user_id = f"{user_name}_{namespace}"

    print(f"\n{'=' * 60}")
    print(f"  Clearing memories for namespace: {namespace}")
    print(f"  User ID: {user_id}")
    print(f"{'=' * 60}")

    list_memory = ListMemory()
    delete_memory = DeleteMemory()

    try:
        result = await list_memory.arun(
            ListMemoryInput(
                user_id=user_id,
                page_size=100,
            )
        )

        if not result or not hasattr(result, "memory_nodes") or not result.memory_nodes:
            print(f"  No memories found for '{namespace}'.")
            return True

        total_count = getattr(result, "total", len(result.memory_nodes))
        print(f"  Found {total_count} memory node(s).")

        if not force:
            confirm = input(f"  Are you sure you want to delete all {total_count} memories? [y/N]: ")
            if confirm.lower() != "y":
                print("  Cancelled.")
                return False

        deleted_count = 0
        while result and hasattr(result, "memory_nodes") and result.memory_nodes:
            for node in result.memory_nodes:
                try:
                    memory_node_id = node.memory_node_id if hasattr(node, "memory_node_id") else str(node)
                    await delete_memory.arun(
                        DeleteMemoryInput(
                            user_id=user_id,
                            memory_node_id=memory_node_id,
                        )
                    )
                    deleted_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to delete memory {memory_node_id}: {e}")

            result = await list_memory.arun(
                ListMemoryInput(
                    user_id=user_id,
                    page_size=100,
                )
            )

        print(f"  Successfully deleted {deleted_count}/{total_count} memories.")
        return deleted_count == total_count

    except Exception as e:
        print(f"  [ERROR] Failed to clear memories: {e}")
        return False
    finally:
        await list_memory.close()
        await delete_memory.close()


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
    parser = argparse.ArgumentParser(description="View and manage long-term memories for TradingScope")
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help=f"Memory namespace to operate on (default: '{DEFAULT_NAMESPACE}')",
    )
    parser.add_argument(
        "--query",
        default="股票交易经验教训",
        help="Search query for memory retrieval (default: '股票交易经验教训')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max number of memories to retrieve (default: 5)",
    )
    parser.add_argument(
        "--user-name",
        default="tradingscope",
        help="User name prefix for memory namespace (default: 'tradingscope')",
    )

    # Add lesson subcommand
    parser.add_argument("--add", action="store_true", help="Add a trading lesson")
    parser.add_argument("--situation", help="Market situation for the lesson")
    parser.add_argument("--decision", help="Trading decision made")
    parser.add_argument("--outcome", help="Actual outcome")
    parser.add_argument("--lesson", help="Lesson learned")

    # Clear memory options
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear memories for the specified namespace",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt (use with --clear)",
    )

    # Review analysis records
    parser.add_argument(
        "--review",
        nargs="?",
        const="all",
        metavar="TICKER",
        help="Review saved analysis records. Use 'all' or omit ticker to show all.",
    )
    parser.add_argument("--date", help="Filter analysis records by date (YYYY-MM-DD)")
    parser.add_argument("--results-dir", help="Path to results directory")

    args = parser.parse_args()

    if args.review is not None:
        ticker = None if args.review.lower() == "all" else args.review
        review_records(ticker=ticker, date=args.date, results_dir=args.results_dir)
    elif args.clear:
        asyncio.run(clear_memories(args.namespace, args.user_name, args.force))
    elif args.add:
        if not all([args.situation, args.decision, args.outcome, args.lesson]):
            parser.error("--situation, --decision, --outcome, --lesson are all required with --add")
        asyncio.run(
            add_lesson(
                namespace=args.namespace,
                situation=args.situation,
                decision=args.decision,
                outcome=args.outcome,
                lesson=args.lesson,
                user_name=args.user_name,
            )
        )
    else:
        asyncio.run(view_memories(args.namespace, args.query, args.top_k, args.user_name))


if __name__ == "__main__":
    main()
