"""Tests for strict two-phase structured agent execution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.test_output_models import _all_outputs


@pytest.mark.asyncio
async def test_runner_returns_concrete_pydantic_output() -> None:
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    expected = _all_outputs()[0]
    formatter_model = SimpleNamespace(
        generate_structured_output=AsyncMock(
            return_value=SimpleNamespace(content=expected.model_dump(mode="json")),
        ),
    )
    runner = StructuredAgentRunner(formatter_model=formatter_model)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: "分析素材"))

    agent = SimpleNamespace(name="MarketAnalyst")
    result = await runner.run(agent, type(expected))

    assert result == expected
    runner._run_analysis.assert_awaited_once_with(agent, None)
    assert formatter_model.generate_structured_output.await_count == 1


@pytest.mark.asyncio
async def test_runner_retries_only_structured_phase() -> None:
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    expected = _all_outputs()[0]
    invalid = {**expected.model_dump(mode="json"), "decision": {}}
    formatter_model = SimpleNamespace(
        generate_structured_output=AsyncMock(
            side_effect=[
                SimpleNamespace(content=invalid),
                SimpleNamespace(content=expected.model_dump(mode="json")),
            ],
        ),
    )
    runner = StructuredAgentRunner(formatter_model=formatter_model)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: "分析素材"))

    result = await runner.run(SimpleNamespace(name="MarketAnalyst"), type(expected), None)

    assert result == expected
    assert runner._run_analysis.await_count == 1
    assert formatter_model.generate_structured_output.await_count == 2
    retry_prompt = formatter_model.generate_structured_output.await_args_list[1].kwargs["messages"][0].get_text_content()
    assert "decision.direction" in retry_prompt


@pytest.mark.asyncio
async def test_runner_retries_when_model_emits_no_structured_tool_call() -> None:
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    expected = _all_outputs()[0]
    formatter_model = SimpleNamespace(
        generate_structured_output=AsyncMock(
            side_effect=[
                RuntimeError("Failed to generate structured output for model."),
                SimpleNamespace(content=expected.model_dump(mode="json")),
            ],
        ),
    )
    runner = StructuredAgentRunner(formatter_model=formatter_model)
    runner._run_analysis = AsyncMock(
        return_value=SimpleNamespace(get_text_content=lambda: "分析素材"),
    )

    result = await runner.run(
        SimpleNamespace(name="MarketAnalyst"),
        type(expected),
        None,
    )

    assert result == expected
    assert runner._run_analysis.await_count == 1
    assert formatter_model.generate_structured_output.await_count == 2


@pytest.mark.asyncio
async def test_runner_fails_after_three_invalid_outputs() -> None:
    from tradingscope.agents.utils.structured_output import (
        StructuredAgentRunner,
        StructuredOutputValidationError,
    )

    expected = _all_outputs()[0]
    formatter_model = SimpleNamespace(
        generate_structured_output=AsyncMock(
            return_value=SimpleNamespace(content={"schema_version": "2.0"}),
        ),
    )
    runner = StructuredAgentRunner(formatter_model=formatter_model, max_validation_attempts=3)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: "分析素材"))

    with pytest.raises(StructuredOutputValidationError, match="MarketAnalyst") as exc_info:
        await runner.run(SimpleNamespace(name="MarketAnalyst"), type(expected), None)

    assert formatter_model.generate_structured_output.await_count == 3
    assert exc_info.value.errors
    assert "ticker: Field required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_runner_retries_generated_output_policy_failure() -> None:
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    expected = _all_outputs()[5]
    invalid = expected.model_dump(mode="json")
    invalid["decision"] = {
        "direction": "bearish",
        "action": "sell",
        "confidence": 0.6,
        "summary": "卖出",
        "reasoning": ["趋势向下"],
    }
    invalid["trade_intent"] = "open_short"
    invalid["position_advice"] = "light"
    invalid["time_stop_days"] = 3
    invalid["price_plan"] = {
        "entry_price": 100.0,
        "entry_price_low": None,
        "entry_price_high": None,
        "target_price": None,
        "stop_loss": 105.0,
        "currency": "USD",
        "invalidation_conditions": ["站上105"],
    }
    valid = {
        **invalid,
        "price_plan": {
            **invalid["price_plan"],
            "entry_price_low": 99.0,
            "entry_price_high": 101.0,
            "target_price": 90.0,
        },
    }
    formatter_model = SimpleNamespace(
        generate_structured_output=AsyncMock(
            side_effect=[
                SimpleNamespace(content=invalid),
                SimpleNamespace(content=valid),
            ],
        ),
    )
    runner = StructuredAgentRunner(formatter_model=formatter_model)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: "分析素材"))

    result = await runner.run(SimpleNamespace(name="Trader"), type(expected))

    assert result.price_plan.target_price == 90.0
    assert formatter_model.generate_structured_output.await_count == 2
    retry_prompt = formatter_model.generate_structured_output.await_args_list[1].kwargs["messages"][0].get_text_content()
    assert "price_plan.target_price" in retry_prompt
