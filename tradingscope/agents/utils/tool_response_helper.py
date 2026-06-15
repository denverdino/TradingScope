import copy
import functools
import os
from typing import Any, Callable, TypeVar

from agentscope import logger
from agentscope.message import TextBlock, ToolResultState
from agentscope.tool import ToolChunk

F = TypeVar("F", bound=Callable[..., Any])


def _to_tool_chunk(text: str) -> ToolChunk:
    """Wrap a plain string result into a ToolChunk for AgentScope 2.0."""
    return ToolChunk(content=[TextBlock(text=text)], state=ToolResultState.SUCCESS)


def agentscope_tool(func: F) -> F:
    """Caching decorator for tool functions. Returns ToolChunk results."""
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> ToolChunk:
        key = (args, tuple(sorted(kwargs.items())))
        debug = os.getenv("TOOL_DEBUG")

        if key in cache:
            if debug:
                logger.info(f"🔧 [TOOL_DEBUG] {func.__name__}({_fmt_args(args, kwargs)}) -> (cached)")
            return _to_tool_chunk(str(copy.deepcopy(cache[key])))

        if debug:
            logger.info(f"🔧 [TOOL_DEBUG] {func.__name__}({_fmt_args(args, kwargs)}) ...")

        try:
            result = func(*args, **kwargs)
            cache[key] = result
            if debug:
                preview = str(result)[:500]
                logger.info(f"🔧 [TOOL_DEBUG] {func.__name__} -> {preview}")
            return _to_tool_chunk(str(copy.deepcopy(result)))
        except Exception as e:
            if debug:
                logger.error(f"🔧 [TOOL_DEBUG] {func.__name__} raised {type(e).__name__}: {e}")
            raise

    return wrapper  # type: ignore


def _fmt_args(args: tuple, kwargs: dict) -> str:
    """Format positional and keyword arguments for debug display."""
    parts = [repr(a) for a in args]
    parts += [f"{k}={v!r}" for k, v in kwargs.items()]
    return ", ".join(parts)
