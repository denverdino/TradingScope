"""Alibaba Cloud compatibility for AgentScope's Responses API model."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from copy import deepcopy
from datetime import datetime
from typing import Any

from agentscope.message import Msg
from agentscope.model import ChatResponse, OpenAIResponseModel, StructuredResponse
from agentscope.tool import ToolChoice
from pydantic import BaseModel


def _inline_local_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline Pydantic ``$defs`` references for DashScope function tools."""
    root = deepcopy(schema)
    definitions = root.get("$defs", {})

    def inline(value: Any) -> Any:
        if isinstance(value, list):
            return [inline(item) for item in value]
        if not isinstance(value, dict):
            return value

        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.removeprefix("#/$defs/")
            definition = definitions.get(name)
            if isinstance(definition, dict):
                merged = deepcopy(definition)
                merged.update({key: item for key, item in value.items() if key != "$ref"})
                return inline(merged)

        return {key: inline(item) for key, item in value.items() if key != "$defs"}

    return inline(root)


def _with_inlined_schema(model: type[BaseModel]) -> type[BaseModel]:
    """Keep Pydantic validation while exposing a ref-free tool schema."""
    schema = _inline_local_refs(model.model_json_schema())

    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del cls, args, kwargs
        return deepcopy(schema)

    return type(
        f"DashScope{model.__name__}",
        (model,),
        {"model_json_schema": classmethod(model_json_schema)},
    )


class DashScopeResponseModel(OpenAIResponseModel):
    """Use Alibaba Cloud Responses with thinking-mode Code Interpreter."""

    async def _parse_stream_response(
        self,
        start_datetime: datetime,
        response: Any,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Raise DashScope stream failures instead of returning empty text."""

        async def checked_events() -> AsyncGenerator[Any, None]:
            async for event in response:
                if event.type == "response.failed":
                    error = getattr(event.response, "error", None)
                    code = getattr(error, "code", None) or "unknown"
                    message = getattr(error, "message", None) or "unknown error"
                    raise RuntimeError(
                        f"DashScope response failed ({code}): {message}",
                    )
                yield event

        async for chunk in super()._parse_stream_response(
            start_datetime,
            checked_events(),
        ):
            yield chunk

    async def _call_api(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        **generate_kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """Translate thinking mode and delegate the request to AgentScope."""
        extra_body = dict(generate_kwargs.pop("extra_body", {}) or {})
        extra_body["enable_thinking"] = self.parameters.thinking_enable
        return await super()._call_api(
            model_name,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            extra_body=extra_body,
            **generate_kwargs,
        )

    def _format_tools(
        self,
        tools: list[dict] | None,
        tool_choice: ToolChoice | None,
    ) -> tuple[list[dict], str | dict | None]:
        """Keep local function tools and append the built-in server tool."""
        formatted_tools, formatted_choice = super()._format_tools(
            tools,
            tool_choice,
        )
        if any(tool.get("type") == "function" and tool.get("function", {}).get("name") == "generate_structured_output" for tool in tools or []):
            return formatted_tools or [], formatted_choice
        if not self.parameters.thinking_enable:
            return formatted_tools or [], formatted_choice
        return [
            *(formatted_tools or []),
            {"type": "code_interpreter"},
        ], formatted_choice

    async def _call_api_with_structured_output(
        self,
        model_name: str,
        messages: list[Msg],
        structured_model: type[BaseModel] | dict,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> StructuredResponse:
        """Generate structured output with DashScope-compatible tool schema."""
        del tool_choice
        if isinstance(structured_model, dict):
            dashscope_model: type[BaseModel] | dict = _inline_local_refs(
                structured_model,
            )
        else:
            dashscope_model = _with_inlined_schema(structured_model)

        return await super()._call_api_with_structured_output(
            model_name=model_name,
            messages=messages,
            structured_model=dashscope_model,
            tool_choice=ToolChoice(
                mode="auto" if self.parameters.thinking_enable else "required",
            ),
            **kwargs,
        )
