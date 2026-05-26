import asyncio
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


async def code_interpreter(code: str, timeout: float = 30) -> ToolChunk:
    """Execute Python code for precise numerical calculations (deviation rates, risk-reward ratios, ATR multiples, etc.).

    Use this tool when you need exact arithmetic that LLMs often miscalculate.
    Write Python code with print() statements to output results.

    Args:
        code: Python code to execute. Must use print() to see output.
        timeout: Maximum execution time in seconds (default 30, capped at 60).

    Returns:
        ToolChunk containing stdout, stderr, and return code from execution.
    """
    timeout = min(timeout, 60)
    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stdout_str = stdout.decode() if stdout else ""
        stderr_str = stderr.decode() if stderr else ""
        result_parts = []
        if stdout_str:
            result_parts.append(f"stdout:\n{stdout_str}")
        if stderr_str:
            result_parts.append(f"stderr:\n{stderr_str}")
        result_parts.append(f"return_code: {proc.returncode}")
        return _to_tool_chunk("\n".join(result_parts))
    except asyncio.TimeoutError:
        return _to_tool_chunk(f"Error: Code execution timed out after {timeout} seconds")
    except Exception as e:
        return _to_tool_chunk(f"Error: {type(e).__name__}: {e}")
