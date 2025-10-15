import functools
from typing import Any, Callable, TypeVar

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

F = TypeVar("F", bound=Callable[..., Any])

def agentscope_tool(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> ToolResponse:
        result = func(*args, **kwargs)
        return ToolResponse(content=[TextBlock(type="text", text=str(result))])
    return wrapper  # type: ignore
