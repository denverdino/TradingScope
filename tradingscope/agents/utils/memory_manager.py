"""Memory Manager for TradingScope agents.

This module provides a centralized manager for creating and managing
ModelStudioLongTermMemory instances for all decision-making agents.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

from agentscope import logger

from .memory import ModelStudioLongTermMemory


class FinancialMemoryManager:
    """Manages ModelStudioLongTermMemory instances for all decision agents.

    This class creates and manages long-term memory instances for the 5 core
    decision-making agents:
    - bull_researcher: Bullish analysis memory
    - bear_researcher: Bearish analysis memory
    - trader: Trading decision memory
    - research_manager: Research synthesis memory
    - risk_manager: Risk assessment memory

    Usage:
        memory_manager = FinancialMemoryManager()

        # Get memory for specific agent
        bull_memory = memory_manager.get_memory("bull_researcher")

        # Use in ReActAgent
        agent = ReActAgent(
            name="BullResearcher",
            long_term_memory=bull_memory,
            long_term_memory_mode="static_control",
        )

        # Cleanup when done
        await memory_manager.close()
    """

    # Agent roles that have memory capability
    AGENT_ROLES = [
        "bull_researcher",
        "bear_researcher",
        "trader",
        "research_manager",
        "risk_manager",
    ]

    def __init__(self, user_name: str = "tradingscope", top_k: int = 5):
        """Initialize FinancialMemoryManager.

        Args:
            user_name: User identifier prefix for memory namespace isolation
            top_k: Default number of memories to retrieve
        """
        self._user_name = user_name
        self._top_k = top_k
        self._memories: Dict[str, ModelStudioLongTermMemory] = {}

        # Create memory instances for each role
        for role in self.AGENT_ROLES:
            self._memories[role] = ModelStudioLongTermMemory(
                agent_name=role,
                user_name=user_name,
                top_k=top_k
            )

        logger.info(f"FinancialMemoryManager created with {len(self._memories)} memory instances")

    async def close(self) -> None:
        """Close all memory instances.

        Properly closes all API connections.
        """
        logger.info("Closing FinancialMemoryManager...")

        # Close all memories concurrently
        close_tasks = [
            memory.close() for memory in self._memories.values()
        ]
        await asyncio.gather(*close_tasks, return_exceptions=True)

        logger.info("FinancialMemoryManager closed")

    def get_memory(self, role: str) -> Optional[ModelStudioLongTermMemory]:
        """Get memory instance for a specific agent role.

        Args:
            role: Agent role name (e.g., "bull_researcher", "trader")

        Returns:
            ModelStudioLongTermMemory instance for the role, or None if not found
        """
        if role not in self._memories:
            logger.warning(f"Unknown agent role: {role}. Available roles: {self.AGENT_ROLES}")
            return None
        return self._memories.get(role)

    @property
    def bull_researcher_memory(self) -> Optional[ModelStudioLongTermMemory]:
        """Get memory for bull researcher agent."""
        return self.get_memory("bull_researcher")

    @property
    def bear_researcher_memory(self) -> Optional[ModelStudioLongTermMemory]:
        """Get memory for bear researcher agent."""
        return self.get_memory("bear_researcher")

    @property
    def trader_memory(self) -> Optional[ModelStudioLongTermMemory]:
        """Get memory for trader agent."""
        return self.get_memory("trader")

    @property
    def research_manager_memory(self) -> Optional[ModelStudioLongTermMemory]:
        """Get memory for research manager agent."""
        return self.get_memory("research_manager")

    @property
    def risk_manager_memory(self) -> Optional[ModelStudioLongTermMemory]:
        """Get memory for risk manager agent."""
        return self.get_memory("risk_manager")

    async def __aenter__(self) -> "FinancialMemoryManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
