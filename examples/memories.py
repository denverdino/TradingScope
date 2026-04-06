#!/usr/bin/env python3
"""Example script to view and manage long-term memories for each agent role.

Usage:
    # View all agent memories
    python examples.memories

    # View memories for a specific role
    python examples.memories --role trader

    # Search memories with a specific query
    python examples.memories --role bull_researcher --query "技术面强但基本面疲软"

    # Add a trading lesson to a role's memory
    python examples.memories --role trader --add \
        --situation "RSI超卖，MACD金叉" \
        --decision "买入" \
        --outcome "盈利8%" \
        --lesson "RSI超卖配合MACD金叉是较强的买入信号"

    # Clear memories for a specific role (requires confirmation)
    python examples.memories --role trader --clear

    # Clear memories for ALL roles (requires confirmation)
    python examples.memories --clear-all

    # Force clear without confirmation (use with caution!)
    python examples.memories --clear-all --force
"""

import argparse
import asyncio

from tradingscope.agents.utils.memory import ModelStudioLongTermMemory
from tradingscope.agents.utils.memory_manager import FinancialMemoryManager

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


async def view_role_memories(role: str, query: str, top_k: int, user_name: str) -> None:
    """Search and display memories for a specific agent role."""
    memory = ModelStudioLongTermMemory(
        agent_name=role,
        user_name=user_name,
        top_k=top_k,
    )

    try:
        print(f"\n{'='*60}")
        print(f"  Agent Role: {role}")
        print(f"  User ID:    {memory.user_id}")
        print(f"  Query:      {query}")
        print(f"  Top K:      {top_k}")
        print(f"{'='*60}")

        result = await memory.retrieve_from_memory(query)
        if result == "Memory system unavailable":
            print("\n  [ERROR] Memory API not available. Check DASHSCOPE_API_KEY.")
        elif result == "No relevant memories found":
            print("\n  No memories found for this role.")
        else:
            print(f"\n{result}")

    finally:
        await memory.close()


async def view_all_memories(query: str, top_k: int, user_name: str) -> None:
    """Search and display memories for all agent roles."""
    print("\nSearching memories for all agent roles...")
    print(f"Query: '{query}'")
    print(f"Top K: {top_k}")

    for role in FinancialMemoryManager.AGENT_ROLES:
        await view_role_memories(role, query, top_k, user_name)


async def add_lesson(
    role: str,
    situation: str,
    decision: str,
    outcome: str,
    lesson: str,
    user_name: str,
) -> None:
    """Add a trading lesson to a specific agent's memory."""
    memory = ModelStudioLongTermMemory(
        agent_name=role,
        user_name=user_name,
    )

    try:
        print(f"\nAdding trading lesson to {role}'s memory...")
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


async def clear_role_memories(role: str, user_name: str, force: bool = False) -> bool:
    """Clear all memories for a specific agent role.

    Args:
        role: Agent role name
        user_name: User name prefix for memory namespace
        force: Skip confirmation if True

    Returns:
        True if cleared successfully, False otherwise
    """
    if not MEMORY_API_AVAILABLE:
        print("  [ERROR] Memory API not available. Check DASHSCOPE_API_KEY.")
        return False

    user_id = f"{user_name}_{role}"

    print(f"\n{'='*60}")
    print(f"  Clearing memories for: {role}")
    print(f"  User ID: {user_id}")
    print(f"{'='*60}")

    # List all memories first
    list_memory = ListMemory()
    delete_memory = DeleteMemory()

    try:
        # Get first page to check total count
        result = await list_memory.arun(ListMemoryInput(
            user_id=user_id,
            page_size=100,
        ))

        if not result or not hasattr(result, 'memory_nodes') or not result.memory_nodes:
            print(f"  No memories found for {role}.")
            return True

        total_count = getattr(result, 'total', len(result.memory_nodes))
        print(f"  Found {total_count} memory node(s).")

        # Confirm deletion
        if not force:
            confirm = input(f"  Are you sure you want to delete all {total_count} memories? [y/N]: ")
            if confirm.lower() != 'y':
                print("  Cancelled.")
                return False

        # Delete all memories by repeatedly fetching page 1 and deleting
        deleted_count = 0
        while result and hasattr(result, 'memory_nodes') and result.memory_nodes:
            for node in result.memory_nodes:
                try:
                    memory_node_id = node.memory_node_id if hasattr(node, 'memory_node_id') else str(node)
                    await delete_memory.arun(DeleteMemoryInput(
                        user_id=user_id,
                        memory_node_id=memory_node_id,
                    ))
                    deleted_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to delete memory {memory_node_id}: {e}")

            # Fetch next batch (always page 1 since previous ones were deleted)
            result = await list_memory.arun(ListMemoryInput(
                user_id=user_id,
                page_size=100,
            ))

        print(f"  Successfully deleted {deleted_count}/{total_count} memories.")
        return deleted_count == total_count

    except Exception as e:
        print(f"  [ERROR] Failed to clear memories: {e}")
        return False
    finally:
        await list_memory.close()
        await delete_memory.close()


async def clear_all_memories(user_name: str, force: bool = False) -> None:
    """Clear memories for all agent roles.

    Args:
        user_name: User name prefix for memory namespace
        force: Skip confirmation if True
    """
    print(f"\n{'#'*60}")
    print("  CLEARING ALL AGENT MEMORIES")
    print(f"  User name: {user_name}")
    print(f"  Roles: {', '.join(FinancialMemoryManager.AGENT_ROLES)}")
    print(f"{'#'*60}")

    if not force:
        confirm = input("\n  WARNING: This will delete ALL memories for ALL roles!\n  Are you sure? [y/N]: ")
        if confirm.lower() != 'y':
            print("  Cancelled.")
            return

    results = {}
    for role in FinancialMemoryManager.AGENT_ROLES:
        # Force=True for individual roles since we already confirmed above
        success = await clear_role_memories(role, user_name, force=True)
        results[role] = success

    # Summary
    print(f"\n{'='*60}")
    print("  CLEAR SUMMARY")
    print(f"{'='*60}")
    for role, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"    {role}: {status}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="View and manage long-term memories for TradingScope agent roles"
    )
    parser.add_argument(
        "--role",
        choices=FinancialMemoryManager.AGENT_ROLES,
        help="Specific agent role to query (default: all roles)",
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
        help="Max number of memories to retrieve per role (default: 5)",
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
        help="Clear memories for a specific role (requires --role)",
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear memories for ALL agent roles",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt (use with --clear or --clear-all)",
    )

    args = parser.parse_args()

    # Handle clear-all first
    if args.clear_all:
        asyncio.run(clear_all_memories(args.user_name, args.force))
    # Handle clear for specific role
    elif args.clear:
        if not args.role:
            parser.error("--role is required when using --clear")
        asyncio.run(clear_role_memories(args.role, args.user_name, args.force))
    # Handle add lesson
    elif args.add:
        if not args.role:
            parser.error("--role is required when adding a lesson")
        if not all([args.situation, args.decision, args.outcome, args.lesson]):
            parser.error("--situation, --decision, --outcome, --lesson are all required with --add")
        asyncio.run(add_lesson(
            role=args.role,
            situation=args.situation,
            decision=args.decision,
            outcome=args.outcome,
            lesson=args.lesson,
            user_name=args.user_name,
        ))
    # View memories
    elif args.role:
        asyncio.run(view_role_memories(args.role, args.query, args.top_k, args.user_name))
    else:
        asyncio.run(view_all_memories(args.query, args.top_k, args.user_name))


if __name__ == "__main__":
    main()
