import copy
import functools
import os
from typing import Any, Callable, TypeVar

from agentscope import logger
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

F = TypeVar("F", bound=Callable[..., Any])


def agentscope_tool(func: F) -> F:
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> ToolResponse:
        key = (args, tuple(sorted(kwargs.items())))
        debug = os.getenv("TOOL_DEBUG")

        if key in cache:
            if debug:
                logger.info(f"🔧 [TOOL_DEBUG] {func.__name__}({_fmt_args(args, kwargs)}) -> (cached)")
            return ToolResponse(content=[TextBlock(type="text", text=str(copy.deepcopy(cache[key])))])

        if debug:
            logger.info(f"🔧 [TOOL_DEBUG] {func.__name__}({_fmt_args(args, kwargs)}) ...")

        try:
            result = func(*args, **kwargs)
            cache[key] = result
            if debug:
                preview = str(result)[:500]
                logger.info(f"🔧 [TOOL_DEBUG] {func.__name__} -> {preview}")
            return ToolResponse(content=[TextBlock(type="text", text=str(copy.deepcopy(result)))])
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
