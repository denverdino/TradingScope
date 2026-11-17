"""Tests for the Alibaba Cloud OpenAI-compatible Responses model."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from agentscope.credential import OpenAICredential
from agentscope.exception import StructuredOutputError
from agentscope.formatter import OpenAIResponseFormatter
from agentscope.message import SystemMsg, TextBlock, ToolCallBlock, UserMsg
from agentscope.model import ChatResponse, ChatUsage, OpenAIResponseModel
from agentscope.tool import ToolChoice
from pydantic import BaseModel

from tests.test_output_models import _all_outputs
from tradingscope.agents.output import FundamentalsAnalystOutput, MarketAnalystOutput
from tradingscope.agents.utils.cache_usage import CacheUsageCollector
from tradingscope.agents.utils.dashscope_response_model import DashScopeResponseModel
from tradingscope.agents.utils.prompt_cache import DashScopeResponseFormatter
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


def test_qwen4_enables_automatic_cache_without_overriding_caller_options() -> None:
    model = _model(thinking_enable=False)
    with patch.object(OpenAIResponseModel, "_call_api", new_callable=AsyncMock) as call_api:
        asyncio.run(model._call_api("qwen4-max", []))
        assert call_api.await_args.kwargs["extra_body"]["prompt_cache_options"] == {"mode": "implicit", "ttl": "1h"}
        options = {"mode": "explicit", "ttl": "5m"}
        asyncio.run(model._call_api("qwen4-max", [], extra_body={"prompt_cache_options": options}))
        assert call_api.await_args.kwargs["extra_body"]["prompt_cache_options"] == options


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
        usage=ChatUsage(input_tokens=1_000, output_tokens=100, cache_input_tokens=800, time=0.2),
    )


def _runner_with_real_structured_api(*, thinking_enable=True, stream=False, usage_collector=None, max_retries=0):
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    model = _model(thinking_enable=thinking_enable, stream=stream)
    model.max_retries = max_retries
    formatter_model = SimpleNamespace(
        generate_structured_output=AsyncMock(wraps=model.generate_structured_output),
    )
    runner = StructuredAgentRunner(formatter_model=formatter_model, max_validation_attempts=3, usage_collector=usage_collector)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: "分析素材"))
    return runner, formatter_model


def _as_raw_stream(response):
    async def chunks():
        yield ChatResponse(content=[], is_last=False)
        yield ChatResponse(content=response.content, usage=response.usage, is_last=False)

    return chunks()


@pytest.mark.parametrize("thinking_enable", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_runner_feeds_back_invalid_execution_range_from_real_structured_api(thinking_enable, stream) -> None:
    collector = CacheUsageCollector()
    runner, formatter_model = _runner_with_real_structured_api(thinking_enable=thinking_enable, stream=stream, usage_collector=collector)

    async def fake_call_api(_model_name, messages, **_kwargs):
        prompt = messages[-1].get_text_content()
        response = _market_tool_call(295.0 if PRICE_RANGE_ERROR in prompt else 310.0)
        return _as_raw_stream(response) if stream else response

    with patch.object(OpenAIResponseModel, "_call_api", new_callable=AsyncMock, side_effect=fake_call_api) as call_api:
        result = asyncio.run(runner.run(SimpleNamespace(name="MarketAnalyst"), MarketAnalystOutput))

    assert result.price_plan.execution_price == 295.0
    assert call_api.await_count == 2
    assert formatter_model.generate_structured_output.await_count == 2
    assert runner._run_analysis.await_count == 1
    assert collector.by_agent["MarketAnalyst/structured_output"] == {
        "calls": 2,
        "input_tokens": 2_000,
        "output_tokens": 200,
        "cache_input_tokens": 1_600,
    }
    assert all(call.kwargs["tool_choice"].mode == ("auto" if thinking_enable else "required") for call in call_api.await_args_list)
    assert all("on_response" not in call.kwargs for call in call_api.await_args_list)


@pytest.mark.parametrize("thinking_enable", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_runner_reports_exhausted_execution_range_errors_from_real_structured_api(thinking_enable, stream) -> None:
    from tradingscope.agents.utils.structured_output import StructuredOutputValidationError

    collector = CacheUsageCollector()
    runner, formatter_model = _runner_with_real_structured_api(thinking_enable=thinking_enable, stream=stream, usage_collector=collector)

    async def fake_call_api(*_args, **_kwargs):
        response = _market_tool_call(310.0)
        return _as_raw_stream(response) if stream else response

    with patch.object(OpenAIResponseModel, "_call_api", new_callable=AsyncMock, side_effect=fake_call_api) as call_api:
        with pytest.raises(StructuredOutputValidationError, match="execution_price must be within the execution price range"):
            asyncio.run(runner.run(SimpleNamespace(name="MarketAnalyst"), MarketAnalystOutput))

    assert call_api.await_count == 3
    assert formatter_model.generate_structured_output.await_count == 3
    assert runner._run_analysis.await_count == 1
    assert collector.by_agent["MarketAnalyst/structured_output"] == {
        "calls": 3,
        "input_tokens": 3_000,
        "output_tokens": 300,
        "cache_input_tokens": 2_400,
    }


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("failure", ["timeout", "rate_limit"])
def test_structured_transport_retries_do_not_consume_validation_attempts(recover, failure) -> None:
    collector = CacheUsageCollector()
    runner, formatter_model = _runner_with_real_structured_api(usage_collector=collector, max_retries=1)
    request = httpx.Request("POST", "https://example.com")
    if failure == "timeout":
        retryable = OpenAIResponseModel._get_retryable_exceptions()[1]
        error = retryable(request=request)
    else:
        retryable = OpenAIResponseModel._get_retryable_exceptions()[2]
        error = retryable("Rate limited", response=httpx.Response(429, request=request), body=None)
    responses = [error, _market_tool_call(295.0) if recover else error]
    with patch.object(OpenAIResponseModel, "_call_api", new_callable=AsyncMock, side_effect=responses) as call_api:
        if recover:
            result = asyncio.run(runner.run(SimpleNamespace(name="MarketAnalyst"), MarketAnalystOutput))
            assert result.price_plan.execution_price == 295.0
        else:
            with pytest.raises(retryable):
                asyncio.run(runner.run(SimpleNamespace(name="MarketAnalyst"), MarketAnalystOutput))

    assert call_api.await_count == 2
    assert formatter_model.generate_structured_output.await_count == 1
    assert runner._run_analysis.await_count == 1
    assert collector.snapshot()["calls"] == (1 if recover else 0)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_shared_structured_model_keeps_concurrent_usage_attribution(stream) -> None:
    collector = CacheUsageCollector()
    runner, _ = _runner_with_real_structured_api(stream=stream, usage_collector=collector)
    both_started = asyncio.Event()
    started = 0

    async def fake_call_api(*_args, **_kwargs):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await both_started.wait()
        response = _market_tool_call(295.0)
        return _as_raw_stream(response) if stream else response

    with patch.object(OpenAIResponseModel, "_call_api", new_callable=AsyncMock, side_effect=fake_call_api):
        await asyncio.gather(*(runner.run(SimpleNamespace(name=name), MarketAnalystOutput) for name in ("MarketA", "MarketB")))

    assert collector.by_agent == {
        f"{name}/structured_output": {
            "calls": 1,
            "input_tokens": 1_000,
            "output_tokens": 100,
            "cache_input_tokens": 800,
        }
        for name in ("MarketA", "MarketB")
    }


@pytest.mark.parametrize("invalid_content", ["no_tool_call", "invalid_json"])
def test_invalid_structured_response_is_counted_without_strategy_fallback(invalid_content) -> None:
    model = _model()
    collector = CacheUsageCollector()
    response = _market_tool_call(295.0)
    if invalid_content == "no_tool_call":
        response.content = [TextBlock(text="No structured object")]
    else:
        response.content[0].input = "not JSON"

    with patch.object(OpenAIResponseModel, "_call_api", new_callable=AsyncMock, return_value=response) as call_api:
        with pytest.raises(StructuredOutputError):
            asyncio.run(
                model.generate_structured_output(
                    messages=[UserMsg(name="user", content="Summarize AAPL")],
                    structured_model=MarketAnalystOutput,
                    on_response=lambda result: collector.record(result, "MarketAnalyst/structured_output"),
                )
            )

    assert call_api.await_count == 1
    assert collector.snapshot()["calls"] == 1
    assert collector.snapshot()["input_tokens"] == 1_000


@pytest.mark.parametrize("stream", [False, True])
def test_structured_response_without_usage_is_not_estimated(stream) -> None:
    model = _model(stream=stream)
    collector = CacheUsageCollector()
    response = _market_tool_call(295.0)
    response.usage = None

    async def fake_call_api(*_args, **_kwargs):
        return _as_raw_stream(response) if stream else response

    with patch.object(OpenAIResponseModel, "_call_api", new_callable=AsyncMock, side_effect=fake_call_api) as call_api:
        result = asyncio.run(
            model.generate_structured_output(
                messages=[UserMsg(name="user", content="Summarize AAPL")],
                structured_model=MarketAnalystOutput,
                on_response=lambda result: collector.record(result, "MarketAnalyst/structured_output"),
            )
        )

    assert result.content["price_plan"]["execution_price"] == 295.0
    assert call_api.await_count == 1
    assert collector.snapshot() == {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_input_tokens": 0}
    assert collector.by_agent == {}


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
        formatter=DashScopeResponseFormatter(),
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


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY") or os.getenv("RUN_DASHSCOPE_INTEGRATION") != "1",
    reason="DASHSCOPE_API_KEY and RUN_DASHSCOPE_INTEGRATION=1 are required",
)
@pytest.mark.parametrize("stream", [False, True])
def test_shared_context_cache_across_roles_live(stream: bool) -> None:
    from uuid import uuid4

    from tradingscope.agents.utils.prompt_cache import build_cacheable_system_prompt

    model = _live_model()
    model.stream = stream
    model.parameters.thinking_enable = False
    model.parameters.max_tokens = 32
    shared_context = f"Cache probe {uuid4()}:\n" + "市场数据：价格100，支撑95，阻力110；成交量稳定。\n" * 100

    async def probe():
        usages = []
        for role in ("BullResearcher", "BearResearcher"):
            response = await model(
                messages=[
                    SystemMsg(
                        name=role,
                        content=build_cacheable_system_prompt(
                            shared_context=shared_context,
                            role_instructions=f"你是{role}，只回复：收到。",
                        ),
                    ),
                    UserMsg(name="DebateOrchestrator", content="只回复：收到。"),
                ],
            )
            if stream:
                async for chunk in response:
                    if chunk.is_last:
                        response = chunk
            assert response.usage is not None
            usages.append(response.usage)
            print(f"{role}: input={response.usage.input_tokens}, cache_read={response.usage.cache_input_tokens}")
        return usages

    usages = asyncio.run(probe())
    assert usages[1].cache_input_tokens >= 1024


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DASHSCOPE_API_KEY") or os.getenv("RUN_DASHSCOPE_INTEGRATION") != "1",
    reason="DASHSCOPE_API_KEY and RUN_DASHSCOPE_INTEGRATION=1 are required",
)
def test_layered_stage_context_cache_live() -> None:
    from uuid import uuid4

    from tradingscope.agents.utils.prompt_cache import CACHE_PREFIX_BREAK, build_cacheable_system_prompt

    model = _live_model()
    model.parameters.thinking_enable = False
    model.parameters.max_tokens = 32
    analysts = f"Analyst probe {uuid4()}:\n" + "分析师证据：价格100，支撑95，阻力110（来源：行情）。\n" * 100
    research = "研究经理证据：主情景区间震荡，失效条件为收盘跌破95（来源：辩论）。\n" * 100
    trader = "交易员证据：持有，止损95，目标110。\n" * 20

    async def probe():
        usages = []
        for role, shared_context in (
            ("Trader", f"{analysts}{CACHE_PREFIX_BREAK}{research}"),
            ("AggressiveRiskDebater", f"{analysts}{CACHE_PREFIX_BREAK}{research}{CACHE_PREFIX_BREAK}{trader}"),
        ):
            response = await model(
                messages=[
                    SystemMsg(
                        name=role,
                        content=build_cacheable_system_prompt(shared_context=shared_context, role_instructions=f"你是{role}，只回复：收到。"),
                    ),
                    UserMsg(name="Orchestrator", content="只回复：收到。"),
                ],
            )
            assert response.usage is not None
            usages.append(response.usage)
            print(f"{role}: input={response.usage.input_tokens}, cache_read={response.usage.cache_input_tokens}")
        return usages

    usages = asyncio.run(probe())
    assert usages[1].cache_input_tokens >= 2048
