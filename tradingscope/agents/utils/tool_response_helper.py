import copy
import functools
from typing import Any, Callable, TypeVar

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

F = TypeVar("F", bound=Callable[..., Any])

def agentscope_tool(func: F) -> F:
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> ToolResponse:
        key = (args, tuple(sorted(kwargs.items())))

        if key in cache:
            return ToolResponse(content=[TextBlock(type="text", text=str(copy.deepcopy(cache[key])))])

        try:
            result = func(*args, **kwargs)
            cache[key] = result
            return ToolResponse(content=[TextBlock(type="text", text=str(copy.deepcopy(result)))])
        except Exception:
            raise

    return wrapper  # type: ignore
