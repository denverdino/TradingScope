"""Tests for the Alibaba Cloud OpenAI-compatible Responses model."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest
from agentscope.credential import OpenAICredential
from agentscope.formatter import OpenAIResponseFormatter
from agentscope.message import TextBlock, ToolCallBlock, UserMsg
from agentscope.model import ChatResponse, OpenAIResponseModel
from agentscope.tool import ToolChoice
from pydantic import BaseModel

from tests.test_output_models import _all_outputs
from tradingscope.agents.output import FundamentalsAnalystOutput, MarketAnalystOutput
from tradingscope.agents.utils.dashscope_response_model import DashScopeResponseModel
from tradingscope.default_config import DEFAULT_CONFIG


class AAPLStructuredResult(BaseModel):
    """Small structured result used to verify the production model boundary."""

    ticker: Literal["AAPL"]
    average_close: float


def _valid_fundamentals_json() -> str:
    return json.dumps(
        {
            "schema_version": "3.0",
            "agent_name": "FundamentalsAnalyst",
            "ticker": "AAPL",
            "analysis_date": "2026-08-12",
            "last_completed_session": "2026-08-12",
            "decision": {
                "direction": "neutral",
                "action": "hold",
                "confidence": 0.5,
                "summary": "数据有限，维持持有。",
                "reasoning": ["关键财务数据缺失"],
            },
            "evidence": [
                {
                    "claim": "财务数据缺失",
                    "supporting_data": "素材未提供收入和利润数据",
                    "source": "用户素材",
                    "as_of_date": "2026-08-12",
                },
            ],
            "limitations": ["缺少收入和利润数据"],
            "company_overview": "素材未提供公司概述。",
            "financial_performance": [],
            "valuation_assessment": "fair",
            "earnings_quality": "数据不足，无法评估。",
            "key_catalysts": [],
            "key_risks": ["数据不足"],
            "price_plan": {
                "execution_price": None,
                "execution_price_low": None,
                "execution_price_high": None,
                "target_price": None,
                "stop_loss": None,
                "currency": "USD",
                "invalidation_conditions": ["获得新财务数据后重新评估"],
            },
        },
        ensure_ascii=False,
    )


def _model(*, thinking_enable: bool = True, stream: bool = False) -> DashScopeResponseModel:
    model = DashScopeResponseModel.__new__(DashScopeResponseModel)
    model.model = "qwen3.8-max"
    model.parameters = DashScopeResponseModel.Parameters(thinking_enable=thinking_enable)
    model.stream = stream
    model.max_retries = 0
    model.retry_delay = 0.0
    model.formatter = OpenAIResponseFormatter()
    return model


def test_model_offers_code_interpreter_without_local_tools() -> None:
    model = _model(thinking_enable=True)

    assert model._format_tools(None, None) == ([{"type": "code_interpreter"}], None)


def test_model_omits_code_interpreter_in_non_thinking_mode() -> None:
    model = _model(thinking_enable=False)

    assert model._format_tools(None, None) == ([], None)


def test_model_keeps_local_function_tools_with_code_interpreter() -> None:
    model = _model(thinking_enable=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_data",
                "description": "Get AAPL stock data",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    formatted_tools, formatted_choice = model._format_tools(
        tools,
        ToolChoice(mode="get_stock_data"),
    )

    assert formatted_tools == [
        {
            "type": "function",
            "name": "get_stock_data",
            "description": "Get AAPL stock data",
            "parameters": {"type": "object", "properties": {}},
        },
        {"type": "code_interpreter"},
    ]
    assert formatted_choice == {"type": "function", "name": "get_stock_data"}


def test_model_passes_thinking_mode_without_discarding_extra_body() -> None:
    model = _model(thinking_enable=False)

    with patch.object(
        OpenAIResponseModel,
        "_call_api",
        new_callable=AsyncMock,
    ) as call_api:
        asyncio.run(
            model._call_api(
                "qwen3.8-max",
                [],
                extra_body={"existing": "value"},
            ),
        )

    assert call_api.await_args.kwargs["extra_body"] == {
        "existing": "value",
        "enable_thinking": False,
    }


def test_model_surfaces_failed_stream_responses() -> None:
    model = _model(thinking_enable=False, stream=True)

    async def failed_events():
        yield SimpleNamespace(
            type="response.failed",
            response=SimpleNamespace(
                error=SimpleNamespace(
                    code="InvalidParameter",
                    message="Normal mode does not support Code interpreter.",
                ),
            ),
        )

    async def consume_stream() -> None:
        async for _chunk in model._parse_stream_response(
            datetime.now(),
            failed_events(),
        ):
            pass

    with pytest.raises(
        RuntimeError,
        match=(
            r"DashScope response failed \(InvalidParameter\): "
            r"Normal mode does not support Code interpreter"
        ),
    ):
        asyncio.run(consume_stream())


def test_model_inlines_nested_schema_for_structured_output() -> None:
    model = _model(thinking_enable=False)
    parent_response = ChatResponse(
        content=[
            ToolCallBlock(
                id="call-1",
                name="generate_structured_output",
                input='{"ticker":"AAPL","average_close":231.63}',
            ),
        ],
        is_last=True,
    )

    with patch.object(
        OpenAIResponseModel,
        "_call_api",
        new_callable=AsyncMock,
        return_value=parent_response,
    ) as call_api:
        response = asyncio.run(
            model.generate_structured_output(
                messages=[UserMsg(name="user", content="Summarize AAPL")],
                structured_model=AAPLStructuredResult,
            ),
        )

    result = AAPLStructuredResult.model_validate(response.content)
    assert result == AAPLStructuredResult(ticker="AAPL", average_close=231.63)
    assert call_api.await_args.kwargs["extra_body"] == {"enable_thinking": False}
    assert call_api.await_args.kwargs["tools"][0]["function"]["name"] == ("generate_structured_output")
    assert call_api.await_args.kwargs["tools"][0]["function"]["parameters"] == (AAPLStructuredResult.model_json_schema())
    assert call_api.await_args.kwargs["tool_choice"] == ToolChoice(mode="required")


def test_model_inlines_pydantic_refs_for_dashscope_tools() -> None:
    model = _model(thinking_enable=False)
    parent_response = ChatResponse(
        content=[
            ToolCallBlock(
                id="call-1",
                name="generate_structured_output",
                input=_valid_fundamentals_json(),
            ),
        ],
        is_last=True,
    )

    with patch.object(
        OpenAIResponseModel,
        "_call_api",
        new_callable=AsyncMock,
        return_value=parent_response,
    ) as call_api:
        asyncio.run(
            model.generate_structured_output(
                messages=[UserMsg(name="user", content="Summarize AAPL")],
                structured_model=FundamentalsAnalystOutput,
            ),
        )

    schema = call_api.await_args.kwargs["tools"][0]["function"]["parameters"]
    assert "$defs" not in schema
    assert schema["properties"]["decision"]["type"] == "object"
    assert schema["properties"]["price_plan"]["type"] == "object"
    price_plan_fields = schema["properties"]["price_plan"]["properties"]
    assert "execution_price_low <= execution_price <= execution_price_high" in price_plan_fields["execution_price"]["description"]
    assert "同时给出或同时留空" in price_plan_fields["execution_price_low"]["description"]
    assert "同时给出或同时留空" in price_plan_fields["execution_price_high"]["description"]
    assert schema["properties"]["evidence"]["items"]["type"] == "object"


def test_model_uses_auto_schema_function_in_thinking_mode() -> None:
    model = _model(thinking_enable=True)
    parent_response = ChatResponse(
        content=[
            ToolCallBlock(
                id="call-1",
                name="generate_structured_output",
                input='{"ticker":"AAPL","average_close":231.63}',
            ),
        ],
        is_last=True,
    )

    with patch.object(
        OpenAIResponseModel,
        "_call_api",
        new_callable=AsyncMock,
        return_value=parent_response,
    ) as call_api:
        asyncio.run(
            model.generate_structured_output(
                messages=[UserMsg(name="user", content="Summarize AAPL")],
                structured_model=AAPLStructuredResult,
            ),
        )

    assert call_api.await_args.kwargs["tool_choice"] == ToolChoice(mode="auto")


def test_model_preserves_pydantic_normalization_with_inlined_schema() -> None:
    model = _model(thinking_enable=False)
    market = _all_outputs()[0].model_dump(mode="json")
    market["weekly_bollinger"] = {
        "upper_band": 340.64,
        "middle_band": 297.07,
        "lower_band": 253.5,
        "signal": "neutral",
    }
    market["weekly_bollinger"] = json.dumps(
        market["weekly_bollinger"],
        ensure_ascii=False,
    )
    parent_response = ChatResponse(
        content=[
            ToolCallBlock(
                id="call-1",
                name="generate_structured_output",
                input=json.dumps(market, ensure_ascii=False),
            ),
        ],
        is_last=True,
    )

    with patch.object(
        OpenAIResponseModel,
        "_call_api",
        new_callable=AsyncMock,
        return_value=parent_response,
    ):
        response = asyncio.run(
            model.generate_structured_output(
                messages=[UserMsg(name="user", content="Summarize AAPL")],
                structured_model=MarketAnalystOutput,
            ),
        )

    result = MarketAnalystOutput.model_validate(response.content)
    assert result.weekly_bollinger is not None


PRICE_RANGE_ERROR = "price_plan: Value error, execution_price must be within the execution price range"


def _market_tool_call(execution_price: float) -> ChatResponse:
    market = _all_outputs()[0].model_dump(mode="json")
    market["price_plan"].update(
        execution_price=execution_price,
        execution_price_low=290.0,
        execution_price_high=300.0,
    )
    return ChatResponse(
        content=[
            ToolCallBlock(
                id="call-price-plan",
                name="generate_structured_output",
                input=json.dumps(market, ensure_ascii=False),
            ),
        ],
        is_last=True,
    )


def _runner_with_real_structured_api():
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    model = _model(thinking_enable=True)
    formatter_model = SimpleNamespace(
        generate_structured_output=AsyncMock(wraps=model.generate_structured_output),
    )
    runner = StructuredAgentRunner(formatter_model=formatter_model, max_validation_attempts=3)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: "分析素材"))
    return runner, formatter_model


def test_runner_feeds_back_invalid_execution_range_from_real_structured_api() -> None:
    runner, formatter_model = _runner_with_real_structured_api()

    async def fake_call_api(_model_name, messages, **_kwargs):
        prompt = messages[-1].get_text_content()
        return _market_tool_call(295.0 if PRICE_RANGE_ERROR in prompt else 310.0)

    with patch.object(OpenAIResponseModel, "_call_api", new_callable=AsyncMock, side_effect=fake_call_api):
        result = asyncio.run(runner.run(SimpleNamespace(name="MarketAnalyst"), MarketAnalystOutput))

    assert result.price_plan.execution_price == 295.0
    assert formatter_model.generate_structured_output.await_count == 2
    assert runner._run_analysis.await_count == 1


def test_runner_reports_exhausted_execution_range_errors_from_real_structured_api() -> None:
    from tradingscope.agents.utils.structured_output import StructuredOutputValidationError

    runner, formatter_model = _runner_with_real_structured_api()

    with patch.object(OpenAIResponseModel, "_call_api", new_callable=AsyncMock, return_value=_market_tool_call(310.0)):
        with pytest.raises(StructuredOutputValidationError, match="execution_price must be within the execution price range"):
            asyncio.run(runner.run(SimpleNamespace(name="MarketAnalyst"), MarketAnalystOutput))

    assert formatter_model.generate_structured_output.await_count == 3


def test_structured_output_does_not_offer_code_interpreter() -> None:
    model = DashScopeResponseModel.__new__(DashScopeResponseModel)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_structured_output",
                "description": "Return AAPL data",
                "parameters": AAPLStructuredResult.model_json_schema(),
            },
        },
    ]

    formatted_tools, formatted_choice = model._format_tools(
        tools,
        ToolChoice(mode="required"),
    )

    assert formatted_tools == [
        {
            "type": "function",
            "name": "generate_structured_output",
            "description": "Return AAPL data",
            "parameters": AAPLStructuredResult.model_json_schema(),
        },
    ]
    assert formatted_choice == "required"


def _live_model() -> DashScopeResponseModel:
    return DashScopeResponseModel(
        credential=OpenAICredential(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=DEFAULT_CONFIG["backend_url"],
        ),
        model=DEFAULT_CONFIG["deep_think_llm"],
        parameters=DashScopeResponseModel.Parameters(thinking_enable=True),
        stream=False,
        formatter=OpenAIResponseFormatter(),
        client_kwargs={
            "default_headers": {
                "x-dashscope-session-cache": "enable",
            },
        },
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY") or os.getenv("RUN_DASHSCOPE_INTEGRATION") != "1",
    reason="DASHSCOPE_API_KEY and RUN_DASHSCOPE_INTEGRATION=1 are required",
)
def test_aapl_code_interpreter_live() -> None:
    model = _live_model()
    response = asyncio.run(
        model(
            messages=[
                UserMsg(
                    name="user",
                    content=(
                        "For AAPL sample closes [229.87, 232.80, 233.33, "
                        "231.59, 230.56], use Python Code Interpreter to "
                        "calculate the arithmetic mean and population standard "
                        "deviation. Return both values."
                    ),
                ),
            ],
        ),
    )

    text = "".join(block.text for block in response.content if isinstance(block, TextBlock))
    assert text.strip()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY") or os.getenv("RUN_DASHSCOPE_INTEGRATION") != "1",
    reason="DASHSCOPE_API_KEY and RUN_DASHSCOPE_INTEGRATION=1 are required",
)
def test_aapl_structured_output_live() -> None:
    model = _live_model()
    response = asyncio.run(
        model.generate_structured_output(
            messages=[
                UserMsg(
                    name="user",
                    content=("For ticker AAPL, calculate the arithmetic mean of sample closes [229.87, 232.80, 233.33, 231.59, 230.56]."),
                ),
            ],
            structured_model=AAPLStructuredResult,
        ),
    )

    result = AAPLStructuredResult.model_validate(response.content)
    assert result.ticker == "AAPL"
    assert result.average_close == pytest.approx(231.63, abs=0.01)
