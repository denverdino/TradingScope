"""Read-only long-term memory wrapper.

Wraps a ModelStudioLongTermMemory instance to provide retrieve-only access.
The record() method is a no-op, preventing agents from writing their outputs
to memory while still allowing them to read Lessons Learned.
"""

from __future__ import annotations

import logging
from typing import Any, List, Sequence, Union

from agentscope.memory import LongTermMemoryBase
from agentscope.message import Msg

logger = logging.getLogger(__name__)


class ReadOnlyLongTermMemory(LongTermMemoryBase):
    """Long-term memory that only supports retrieval (no recording).

    Used with static_control mode in ReActAgent to allow agents to
    retrieve Lessons Learned from a shared memory namespace without
    writing their own outputs back.
    """

    def __init__(self, wrapped: LongTermMemoryBase) -> None:
        super().__init__()
        self._wrapped = wrapped

    async def retrieve(
        self,
        msg: Union[Msg, List[Msg], None],
        **kwargs: Any,
    ) -> str:
        """Delegate retrieval to the wrapped memory instance."""
        return await self._wrapped.retrieve(msg, **kwargs)

    async def record(
        self,
        msgs: Union[Msg, Sequence[Union[Msg, None]], None],
        **kwargs: Any,
    ) -> None:
        """No-op: do not record agent outputs to memory."""
        logger.debug("ReadOnlyLongTermMemory.record() called — skipped (read-only)")

    async def close(self) -> None:
        """No-op: lifecycle managed by FinancialMemoryManager."""
