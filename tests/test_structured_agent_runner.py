"""Tests for strict two-phase structured agent execution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agentscope.exception import StructuredOutputError, ToolJSONDecodeError
from agentscope.model import ChatUsage

from tests.test_output_models import _all_outputs
from tradingscope.agents.utils.cache_usage import CacheUsageCollector


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
@pytest.mark.parametrize("output_index", range(7))
async def test_conversion_requests_concise_field_assignment_on_every_attempt_without_losing_draft(output_index):
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    expected = _all_outputs()[output_index]
    if output_index in (5, 6):
        data = expected.model_dump(mode="json")
        data.update(trade_intent="reduce_long", position_advice="light", time_stop_days=5)
        data["decision"].update(direction="bullish", action="sell")
        data["price_plan"].update(execution_price=192.5, execution_price_low=191, execution_price_high=193, target_price=200, stop_loss=188)
        # The prompt budget is not a schema validator or a hard truncation policy.
        data["evidence"] = [
            {"claim": "价格结构", "supporting_data": "支撑188", "source": "synthetic prices", "as_of_date": "2026-07-23"},
            {"claim": "成交量", "supporting_data": "均量比0.85", "source": "synthetic volume", "as_of_date": "2026-07-23"},
            {"claim": "短期催化剂", "supporting_data": "订单增加", "source": "https://example.test/orders?id=7&source=raw%2Ffeed", "as_of_date": "2026-07-24"},
            {"claim": "最强反证", "supporting_data": "监管尚未解除", "source": "https://example.test/regulatory#status", "as_of_date": "2026-07-24"},
        ]
        data["price_plan"]["invalidation_conditions"] = ["支撑失守", "报价失真", "流动性不足", "监管状态变化"]
        if output_index == 5:
            data["execution_steps"] = ["核验报价", "核验仓位", "减仓20%", "保留防守单"]
        else:
            data["risk_control_measures"] = ["限制仓位", "禁追价", "保留止损", "核验事件状态"]
        expected = type(expected).model_validate(data)
    draft = (
        "RPLY，2026-07-24；最近完成交易日2026-07-23；USD；"
        "原始URL https://example.test/article?id=7&source=raw%2Ffeed#facts；报价192.50；"
        "核心证据1、2、3；最强反证：监管尚未解除；撤回条件：支撑失守。"
        "OHLC 190/194/188/192.50；PCR（OI）0.88；Max Pain非唯一区间120–155。"
        "trade_intent=reduce_long；方向bullish；减仓20%；执行191–193；目标200；止损188；"
        "time_stop_days=5；执行步骤1、2、3、4；风险控制1、2、3、4；失效条件1、2、3、4。"
    )
    formatter = SimpleNamespace(
        generate_structured_output=AsyncMock(
            side_effect=[SimpleNamespace(content={"schema_version": "3.0"}), SimpleNamespace(content=expected.model_dump(mode="json"))],
        ),
    )
    runner = StructuredAgentRunner(formatter)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: draft))

    # A role alias must not affect instructions selected by the public output type.
    result = await runner.run(SimpleNamespace(name="SyntheticAlias"), type(expected))

    assert result == expected
    runner._run_analysis.assert_awaited_once()
    assert formatter.generate_structured_output.await_count == 2
    for call in formatter.generate_structured_output.await_args_list:
        text = call.kwargs["messages"][0].get_text_content()
        assert draft in text
        for required in (
            "结论先行",
            "最多3条独立关键证据",
            "原始URL不得改写",
            "字段分工",
            "保留所有公共JSON字段",
            "不得添加素材中不存在的事实",
            "股票身份、日期、货币、行情引用和原始数字",
            "不限制执行步骤、风险控制和失效条件",
            "最强反证",
            "time_stop_days",
            "不得强制改为1",
        ):
            assert required in text
        assert call.kwargs["structured_model"] is type(expected)
        if output_index == 0:
            assert "支撑/阻力位弃用时保留有效PCR" in text
            assert "非唯一Max Pain只保留区间解释" in text
        if output_index == 3:
            assert "独立社交采样缺失只在 data_quality 中集中说明一次" in text
            assert "limitations 只写其他具体限制" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [False, True])
async def test_social_conversion_receives_exclusive_data_quality_instructions(retry: bool) -> None:
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    expected = _all_outputs()[3]
    draft = "data_quality：未获取独立社交平台样本，仅采用新闻情绪代理信号。\nlimitations：空头持仓数据滞后三周。"
    responses = [SimpleNamespace(content=expected.model_dump(mode="json"))]
    if retry:
        responses.insert(0, SimpleNamespace(content={"schema_version": "3.0"}))
    formatter_model = SimpleNamespace(generate_structured_output=AsyncMock(side_effect=responses))
    runner = StructuredAgentRunner(formatter_model=formatter_model)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: draft))

    result = await runner.run(SimpleNamespace(name="SocialMediaAnalyst"), type(expected))

    assert result == expected
    runner._run_analysis.assert_awaited_once()
    for call in formatter_model.generate_structured_output.await_args_list:
        assert draft in call.kwargs["messages"][0].get_text_content()
        properties = call.kwargs["structured_model"].model_json_schema()["properties"]
        assert "独立社交样本缺失只在本字段说明一次" in properties["data_quality"].get("description", "")
        description = properties["limitations"].get("description", "")
        assert "不重复 data_quality 中的独立社交样本缺失" in description
        assert "仅保留其他限制" in description
        assert "为空列表 []" in description
        assert "合并表述" in description


@pytest.mark.asyncio
async def test_runner_records_structured_output_usage() -> None:
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    expected = _all_outputs()[0]
    collector = CacheUsageCollector()
    response = SimpleNamespace(
        content=expected.model_dump(mode="json"),
        usage=ChatUsage(input_tokens=1_000, output_tokens=100, cache_input_tokens=800, time=0.2),
    )

    async def generate_structured_output(*, on_response, **_kwargs):
        on_response(response)
        return response

    formatter_model = SimpleNamespace(generate_structured_output=AsyncMock(side_effect=generate_structured_output))
    runner = StructuredAgentRunner(formatter_model=formatter_model, usage_collector=collector)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: "分析素材"))

    await runner.run(SimpleNamespace(name="MarketAnalyst"), type(expected))

    assert collector.snapshot()["cache_input_tokens"] == 800
    assert collector.by_agent["MarketAnalyst/structured_output"] == {
        "calls": 1,
        "input_tokens": 1_000,
        "output_tokens": 100,
        "cache_input_tokens": 800,
    }


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
async def test_runner_retries_when_tool_arguments_are_invalid_json() -> None:
    """A malformed structured tool call must not abort the workflow."""
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    expected = _all_outputs()[0]
    formatter_model = SimpleNamespace(
        generate_structured_output=AsyncMock(
            side_effect=[
                ToolJSONDecodeError("Extra data: line 1 column 527"),
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
    )

    assert result == expected
    assert runner._run_analysis.await_count == 1
    assert formatter_model.generate_structured_output.await_count == 2


@pytest.mark.asyncio
async def test_runner_retries_wrapped_structured_output_error_without_validation_cause() -> None:
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    expected = _all_outputs()[0]
    formatter_model = SimpleNamespace(
        generate_structured_output=AsyncMock(
            side_effect=[
                StructuredOutputError("Failed to generate structured output for model."),
                SimpleNamespace(content=expected.model_dump(mode="json")),
            ],
        ),
    )
    runner = StructuredAgentRunner(formatter_model=formatter_model)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: "分析素材"))

    result = await runner.run(SimpleNamespace(name="MarketAnalyst"), type(expected))

    assert result == expected
    assert formatter_model.generate_structured_output.await_count == 2
    retry_prompt = formatter_model.generate_structured_output.await_args_list[1].kwargs["messages"][0].get_text_content()
    assert "上次未生成符合 schema 的结构化对象" in retry_prompt


@pytest.mark.asyncio
async def test_runner_fails_after_three_invalid_outputs() -> None:
    from tradingscope.agents.utils.structured_output import (
        StructuredAgentRunner,
        StructuredOutputValidationError,
    )

    expected = _all_outputs()[0]
    formatter_model = SimpleNamespace(
        generate_structured_output=AsyncMock(
            return_value=SimpleNamespace(content={"schema_version": "3.0"}),
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
        "execution_price": 100.0,
        "execution_price_low": None,
        "execution_price_high": None,
        "target_price": None,
        "stop_loss": 105.0,
        "currency": "USD",
        "invalidation_conditions": ["站上105"],
    }
    valid = {
        **invalid,
        "price_plan": {
            **invalid["price_plan"],
            "execution_price_low": 99.0,
            "execution_price_high": 101.0,
            "target_price": 90.0,
        },
    }
    collector = CacheUsageCollector()
    responses = iter(
        [
            SimpleNamespace(content=output, usage=ChatUsage(input_tokens=1_000, output_tokens=100, cache_input_tokens=800, time=0.2))
            for output in (invalid, valid)
        ]
    )

    async def generate_structured_output(*, on_response, **_kwargs):
        response = next(responses)
        on_response(response)
        return response

    formatter_model = SimpleNamespace(generate_structured_output=AsyncMock(side_effect=generate_structured_output))
    runner = StructuredAgentRunner(formatter_model=formatter_model, usage_collector=collector)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: "分析素材"))

    result = await runner.run(SimpleNamespace(name="Trader"), type(expected))

    assert result.price_plan.target_price == 90.0
    assert formatter_model.generate_structured_output.await_count == 2
    assert collector.by_agent["Trader/structured_output"] == {
        "calls": 2,
        "input_tokens": 2_000,
        "output_tokens": 200,
        "cache_input_tokens": 1_600,
    }
    retry_prompt = formatter_model.generate_structured_output.await_args_list[1].kwargs["messages"][0].get_text_content()
    assert "price_plan.target_price" in retry_prompt
    assert result.time_stop_days == 3


@pytest.mark.asyncio
async def test_runner_corrects_multi_session_scenario_without_regenerating_draft() -> None:
    from tradingscope.agents.utils.structured_output import StructuredAgentRunner

    expected = _all_outputs()[4]
    valid = expected.model_dump(mode="json")
    invalid = expected.model_dump(mode="json")
    invalid["price_scenarios"][0]["timeframe_days"] = 5
    formatter_model = SimpleNamespace(
        generate_structured_output=AsyncMock(side_effect=[SimpleNamespace(content=invalid), SimpleNamespace(content=valid)]),
    )
    runner = StructuredAgentRunner(formatter_model=formatter_model)
    runner._run_analysis = AsyncMock(return_value=SimpleNamespace(get_text_content=lambda: "下一交易日的研究素材"))

    result = await runner.run(SimpleNamespace(name="ResearchManager"), type(expected))

    assert result.price_scenarios[0].timeframe_days == 1
    assert formatter_model.generate_structured_output.await_count == 2
    runner._run_analysis.assert_awaited_once()
    retry_prompt = formatter_model.generate_structured_output.await_args_list[1].kwargs["messages"][0].get_text_content()
    assert "price_scenarios.0.timeframe_days" in retry_prompt
