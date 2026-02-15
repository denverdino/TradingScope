"""Financial Long-term Memory using Model Studio Memory API.

This module provides a LongTermMemoryBase implementation using Alibaba Cloud
Model Studio's memory API for storing and retrieving financial situation
memories, enabling agents to learn from historical trading experiences.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, Sequence, Union

from agentscope import logger
from agentscope.memory import LongTermMemoryBase
from agentscope.message import Msg

from .summarize import summarize_for_memory

# Module-level logger for debug output (controlled by MEMORY_DEBUG env var)
_debug_logger = logging.getLogger(__name__)

try:
    from agentscope_runtime.tools.modelstudio_memory import (
        AddMemory,
        AddMemoryInput,
        Message,
        SearchMemory,
        SearchMemoryInput,
    )
    MEMORY_API_AVAILABLE = True
except ImportError:
    MEMORY_API_AVAILABLE = False
    logger.warning("Model Studio Memory API not available. Memory features will be disabled.")


class ModelStudioLongTermMemory(LongTermMemoryBase):
    """Long-term memory implementation using Model Studio Memory API.

    This class integrates with AgentScope's memory system and uses
    Alibaba Cloud Model Studio's long-term memory API for persistent
    storage and semantic retrieval of trading experiences.

    Supports both static_control and agent_control modes:
    - static_control: Use record() and retrieve() for developer-controlled operations
    - agent_control: Use record_to_memory() and retrieve_from_memory() as tool functions

    Usage:
        memory = ModelStudioLongTermMemory(
            agent_name="bull_researcher",
            user_name="tradingscope"
        )

        # In ReActAgent
        agent = ReActAgent(
            name="BullResearcher",
            sys_prompt="...",
            model=model,
            memory=InMemoryMemory(),
            long_term_memory=memory,
            long_term_memory_mode="static_control",
        )
    """

    def __init__(
        self,
        agent_name: str,
        user_name: str = "tradingscope",
        top_k: int = 5,
    ):
        """Initialize ModelStudioLongTermMemory.

        Args:
            agent_name: Name of the agent using this memory
            user_name: User identifier for memory namespace isolation
            top_k: Default number of memories to retrieve
        """
        super().__init__()
        self.agent_name = agent_name
        self.user_name = user_name
        self.user_id = f"{user_name}_{agent_name}"
        self.top_k = top_k

        self._add_memory: Optional[AddMemory] = None
        self._search_memory: Optional[SearchMemory] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> bool:
        """Ensure memory API is initialized."""
        if self._initialized:
            return True

        async with self._init_lock:
            if self._initialized:
                return True

            if not MEMORY_API_AVAILABLE:
                logger.warning(f"[{self.user_id}] Memory API not available")
                return False

            try:
                self._add_memory = AddMemory()
                self._search_memory = SearchMemory()
                self._initialized = True
                logger.info(f"[{self.user_id}] Long-term memory initialized")
                return True
            except Exception as e:
                logger.warning(f"[{self.user_id}] Failed to initialize memory API: {e}")
                return False

    async def close(self) -> None:
        """Close memory API connections."""
        if self._add_memory:
            try:
                await self._add_memory.close()
            except Exception as e:
                logger.debug(f"[{self.user_id}] Error closing add_memory: {e}")
            self._add_memory = None

        if self._search_memory:
            try:
                await self._search_memory.close()
            except Exception as e:
                logger.debug(f"[{self.user_id}] Error closing search_memory: {e}")
            self._search_memory = None

        self._initialized = False
        logger.info(f"[{self.user_id}] Long-term memory closed")

    # ============ Raw API Helper ============

    # Model Studio custom_content field has a 512-character limit.
    # Use 500 as safe limit to account for potential edge cases with
    # certain Unicode characters or API overhead.
    _MAX_CONTENT_LENGTH = 500

    async def _add_memory_raw(self, custom_content: str) -> bool:
        """Store content directly via custom_content API field.

        The Model Studio Memory API's `messages` field uses AI inference
        to extract memories, which often returns 0 nodes for complex
        analytical content. The `custom_content` field bypasses inference
        and stores the content directly as a memory node.

        When content exceeds the 512-char API limit, an LLM is used to
        generate a concise summary that preserves key stock analysis
        insights. Falls back to chunk-based splitting if summarization
        is unavailable.

        Args:
            custom_content: Content string to store directly

        Returns:
            True if at least one memory node was created
        """
        if not custom_content:
            return False

        # Use LLM summarization to compress long content into a single
        # memory node, preserving key analytical insights.
        if len(custom_content) > self._MAX_CONTENT_LENGTH:
            custom_content = await summarize_for_memory(
                custom_content, max_chars=self._MAX_CONTENT_LENGTH,
            )

        chunks = self._split_content(custom_content)
        total_saved = 0

        for chunk in chunks:
            payload = {
                "user_id": self.user_id,
                "custom_content": chunk,
            }
            url = self._add_memory.config.get_add_memory_url()
            _debug_logger.debug(
                "[%s] AddMemory request: url=%s, user_id=%s, "
                "chunk_len=%d, chunk_preview=%.100r",
                self.user_id, url, self.user_id, len(chunk), chunk,
            )
            try:
                result = await self._add_memory._request(
                    "POST",
                    url,
                    json=payload,
                )
                _debug_logger.debug(
                    "[%s] AddMemory response: %s",
                    self.user_id, result,
                )
                total_saved += len(result.get("memory_nodes", []))
            except Exception as e:
                logger.warning(
                    f"[{self.user_id}] Failed to store chunk "
                    f"({len(chunk)} chars): {e}",
                )

        return total_saved > 0

    @classmethod
    def _split_content(cls, content: str) -> list[str]:
        """Split content into chunks that fit within the API limit.

        Splits at paragraph boundaries (double newlines) first. If a
        single paragraph still exceeds the limit, it is split at
        single newlines. As a last resort, hard-splits at the limit.

        Args:
            content: Full content string

        Returns:
            List of content chunks, each <= _MAX_CONTENT_LENGTH chars
        """
        limit = cls._MAX_CONTENT_LENGTH
        if len(content) <= limit:
            return [content]

        chunks: list[str] = []
        # Split on double newlines (paragraph boundaries)
        paragraphs = content.split("\n\n")
        current = ""

        for para in paragraphs:
            candidate = f"{current}\n\n{para}" if current else para

            if len(candidate) <= limit:
                current = candidate
            else:
                # Flush current if non-empty
                if current:
                    chunks.append(current)
                    current = ""

                # If this paragraph alone fits, start a new chunk
                if len(para) <= limit:
                    current = para
                else:
                    # Paragraph too long, split on single newlines
                    lines = para.split("\n")
                    for line in lines:
                        line_candidate = (
                            f"{current}\n{line}" if current else line
                        )
                        if len(line_candidate) <= limit:
                            current = line_candidate
                        else:
                            if current:
                                chunks.append(current)
                            # Hard-split if a single line exceeds limit
                            if len(line) <= limit:
                                current = line
                            else:
                                for i in range(0, len(line), limit):
                                    chunks.append(line[i : i + limit])
                                current = ""

        if current:
            chunks.append(current)

        return chunks

    # ============ Static Control Mode Methods ============

    async def record(
        self,
        msgs: Union[Msg, Sequence[Union[Msg, None]], None],
        **kwargs: Any,
    ) -> None:
        """Record messages to long-term memory (static_control mode).

        Called automatically by ReActAgent at the end of reply() in
        static_control mode. The input list may contain None values
        (e.g. when agent is called with msg=None).

        Uses custom_content API to store the agent's final reply directly,
        since the messages-based inference often fails to extract memory
        nodes from complex analytical content.

        Args:
            msgs: Single message, sequence of messages (may contain None), or None
        """
        if not await self._ensure_initialized():
            return

        if msgs is None:
            return

        if isinstance(msgs, Msg):
            msgs = [msgs]

        # Filter out None values - ReActAgent may include None in the list
        msg_list = [m for m in msgs if m is not None and isinstance(m, Msg)]

        if not msg_list:
            return

        try:
            # Extract assistant replies (the agent's actual output)
            assistant_contents = []
            for msg in msg_list:
                if msg.role == "assistant":
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if content:
                        assistant_contents.append(content)

            if not assistant_contents:
                # Fallback: use all message content if no assistant messages
                for msg in msg_list:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if content:
                        assistant_contents.append(content)

            if not assistant_contents:
                return

            # Use custom_content to store the agent's output directly
            # This bypasses the API's inference which often returns 0 nodes
            # for complex analytical content
            custom_content = "\n\n".join(assistant_contents)

            saved = await self._add_memory_raw(custom_content)
            if saved:
                logger.info(f"[{self.user_id}] Recorded memory to long-term storage")
            else:
                logger.warning(f"[{self.user_id}] API returned 0 memory nodes")

        except Exception as e:
            logger.warning(f"[{self.user_id}] Failed to record to memory: {e}")

    async def retrieve(
        self,
        msg: Union[Msg, List[Msg], None],
        **kwargs: Any,
    ) -> str:
        """Retrieve relevant memories based on the input message (static_control mode).

        Called automatically by ReActAgent at the beginning of reply() in
        static_control mode. May receive a single Msg, a list of Msgs, or None.

        Args:
            msg: Message(s) to use as query for retrieval, or None

        Returns:
            Formatted string containing relevant memories, or empty string
        """
        if not await self._ensure_initialized():
            return ""

        if msg is None:
            return ""

        try:
            # Normalize to list of Msg
            if isinstance(msg, Msg):
                msg_list = [msg]
            elif isinstance(msg, list):
                msg_list = [m for m in msg if m is not None and isinstance(m, Msg)]
            else:
                return ""

            if not msg_list:
                return ""

            # Use the last message's content as the search query
            last_msg = msg_list[-1]
            content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

            if not content:
                return ""

            search_input = SearchMemoryInput(
                user_id=self.user_id,
                messages=[Message(role="user", content=content)],
                top_k=self.top_k
            )
            _debug_logger.debug(
                "[%s] SearchMemory request: user_id=%s, top_k=%d, "
                "query_len=%d, query_preview=%.100r",
                self.user_id, self.user_id, self.top_k, len(content), content,
            )

            result = await self._search_memory.arun(search_input)

            _debug_logger.debug(
                "[%s] SearchMemory response: %s",
                self.user_id, result,
            )

            if not result or not hasattr(result, 'memory_nodes') or not result.memory_nodes:
                return ""

            # Format memories for prompt injection
            memories = []
            for node in result.memory_nodes:
                node_content = node.content if hasattr(node, 'content') else str(node)
                if node_content:
                    memories.append(node_content)

            if not memories:
                return ""

            formatted = self._format_memories(memories)
            logger.info(f"[{self.user_id}] Retrieved {len(memories)} memories")
            return formatted

        except Exception as e:
            logger.warning(f"[{self.user_id}] Failed to retrieve memories: {e}")
            return ""

    # ============ Agent Control Mode Methods ============

    async def record_to_memory(self, thinking: str, content: str) -> str:
        """Record content to memory (agent_control mode, used as tool function).

        Args:
            thinking: Agent's reasoning for recording this memory
            content: Content to record to memory

        Returns:
            Status message indicating success or failure
        """
        if not await self._ensure_initialized():
            return "Memory system unavailable"

        try:
            custom_content = f"记忆原因：{thinking}\n{content}"
            saved = await self._add_memory_raw(custom_content)

            if saved:
                logger.debug(f"[{self.user_id}] Agent recorded memory: {content[:100]}...")
                return "Successfully recorded to long-term memory"
            else:
                return "Memory API returned no nodes"

        except Exception as e:
            logger.warning(f"[{self.user_id}] Failed to record memory: {e}")
            return f"Failed to record memory: {str(e)}"

    async def retrieve_from_memory(self, keywords: str) -> str:
        """Retrieve memories based on keywords (agent_control mode, used as tool function).

        Args:
            keywords: Keywords to search for in memory

        Returns:
            Retrieved memories formatted as string
        """
        if not await self._ensure_initialized():
            return "Memory system unavailable"

        try:
            result = await self._search_memory.arun(SearchMemoryInput(
                user_id=self.user_id,
                messages=[Message(role="user", content=keywords)],
                top_k=self.top_k
            ))

            if not result or not hasattr(result, 'memory_nodes') or not result.memory_nodes:
                return "No relevant memories found"

            memories = []
            for node in result.memory_nodes:
                node_content = node.content if hasattr(node, 'content') else str(node)
                if node_content:
                    memories.append(node_content)

            if not memories:
                return "No relevant memories found"

            formatted = self._format_memories(memories)
            logger.debug(f"[{self.user_id}] Agent retrieved {len(memories)} memories")
            return formatted

        except Exception as e:
            logger.warning(f"[{self.user_id}] Failed to retrieve memories: {e}")
            return f"Failed to retrieve memories: {str(e)}"

    # ============ Helper Methods ============

    def _format_memories(self, memories: List[str]) -> str:
        """Format retrieved memories for prompt injection.

        Args:
            memories: List of memory content strings

        Returns:
            Formatted string for prompt injection
        """
        if not memories:
            return ""

        lines = ["# 历史经验教训（来自长期记忆）\n"]
        lines.append("基于过去类似情况的经验：\n")

        for i, memory in enumerate(memories, 1):
            lines.append(f"{i}. {memory}\n")

        lines.append("\n请在分析时考虑这些历史经验教训。")

        return "\n".join(lines)

    async def add_trading_lesson(
        self,
        situation: str,
        decision: str,
        outcome: str,
        lesson: str
    ) -> bool:
        """Add a trading lesson to memory (for future reflection loop).

        Args:
            situation: Market conditions at decision time
            decision: What action was taken
            outcome: What actually happened
            lesson: What should be done differently

        Returns:
            True if successfully recorded, False otherwise
        """
        if not await self._ensure_initialized():
            return False

        try:
            content = (
                f"交易情况回顾：\n"
                f"市场情况：{situation}\n"
                f"交易决策：{decision}\n"
                f"实际结果：{outcome}\n"
                f"经验教训：{lesson}"
            )
            saved = await self._add_memory_raw(content)

            if saved:
                logger.info(f"[{self.user_id}] Added trading lesson to memory")
            return saved

        except Exception as e:
            logger.warning(f"[{self.user_id}] Failed to add trading lesson: {e}")
            return False
