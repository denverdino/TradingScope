"""Memory Manager for TradingScope agents.

This module provides a centralized manager for the shared Lessons Learned
memory namespace. Decision agents receive a read-only wrapper so they can
retrieve past lessons without recording their own outputs.
"""

from __future__ import annotations

from typing import Optional

from agentscope import logger

from .memory import ModelStudioLongTermMemory
from .readonly_memory import ReadOnlyLongTermMemory


class FinancialMemoryManager:
    """Manages a shared Lessons Learned memory namespace.

    The evaluation process writes scored lessons into the ``lessons_learned``
    namespace.  Decision agents receive a :class:`ReadOnlyLongTermMemory`
    wrapper that delegates retrieval to the same namespace but silently
    drops any record() calls.

    Usage:
        memory_manager = FinancialMemoryManager()

        # For agents (read-only access to lessons)
        agent = ReActAgent(
            name="Trader",
            long_term_memory=memory_manager.get_readonly_memory(),
            long_term_memory_mode="static_control",
        )

        # For evaluation process (write access)
        await memory_manager.lessons_memory.add_reflection_lesson(...)

        # Cleanup
        await memory_manager.close()
    """

    def __init__(self, user_name: str = "tradingscope", top_k: int = 5):
        self._user_name = user_name
        self._top_k = top_k
        self._lessons_memory = ModelStudioLongTermMemory(
            agent_name="lessons_learned",
            user_name=user_name,
            top_k=top_k,
        )
        self._readonly_memory: Optional[ReadOnlyLongTermMemory] = None

        logger.info("FinancialMemoryManager created with shared lessons_learned namespace")

    @property
    def lessons_memory(self) -> ModelStudioLongTermMemory:
        """Raw lessons memory (for evaluation process to write lessons)."""
        return self._lessons_memory

    def get_readonly_memory(self) -> ReadOnlyLongTermMemory:
        """Read-only wrapper for agents (retrieve lessons, no recording)."""
        if self._readonly_memory is None:
            self._readonly_memory = ReadOnlyLongTermMemory(self._lessons_memory)
        return self._readonly_memory

    async def close(self) -> None:
        """Close memory API connections."""
        logger.info("Closing FinancialMemoryManager...")
        await self._lessons_memory.close()
        logger.info("FinancialMemoryManager closed")

    async def __aenter__(self) -> "FinancialMemoryManager":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
