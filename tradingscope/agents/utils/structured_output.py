"""Strict two-phase execution for typed agent outputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Generic, TypeVar

from agentscope.exception import StructuredOutputError, ToolJSONDecodeError
from agentscope.message import Msg, UserMsg
from pydantic import ValidationError

from tradingscope.agents.output import AgentOutputBase, MarketAnalystOutput, SocialMediaAnalystOutput

from .agent_utils import call_agent_with_retry
from .decision_policy import CONCISE_ANALYSIS_OUTPUT_POLICY
from .decision_validation import GeneratedOutputPolicyError, validate_generated_output
from .local_audit import get_audit_recorder, redact

OutputT = TypeVar("OutputT", bound=AgentOutputBase)
logger = logging.getLogger(__name__)


def _format_validation_errors(errors: list[dict[str, object]]) -> list[str]:
    lines = []
    for error in errors:
        location = ".".join(str(part) for part in error.get("loc", ()))
        lines.append(f"{location or '<root>'}: {error.get('msg', 'validation failed')}")
    return lines


class StructuredOutputValidationError(RuntimeError):
    """Raised when structured finalization exhausts validation attempts."""

    def __init__(self, agent_name: str, errors: list[dict[str, object]]) -> None:
        self.agent_name = agent_name
        self.errors = errors
        details = "; ".join(_format_validation_errors(errors))
        message = f"{agent_name} failed structured output validation"
        if details:
            message = f"{message}: {details}"
        super().__init__(message)


class StructuredAgentRunner(Generic[OutputT]):
    """Run an agent draft once, then validate its structured final output."""

    def __init__(self, formatter_model, max_validation_attempts: int = 3, usage_collector=None) -> None:
        if max_validation_attempts < 1:
            raise ValueError("max_validation_attempts must be at least 1")
        self._formatter_model = formatter_model
        self._max_validation_attempts = max_validation_attempts
        self._usage_collector = usage_collector

    async def _run_analysis(self, agent, prompt: Msg | None) -> Msg:
        return await call_agent_with_retry(agent, prompt)

    async def run(
        self,
        agent,
        output_model: type[OutputT],
        prompt: Msg | None = None,
        reference_outputs: Sequence[AgentOutputBase] = (),
    ) -> OutputT:
        draft = await self._run_analysis(agent, prompt)
        draft_text = draft.get_text_content()
        audit = get_audit_recorder()
        if audit is not None:
            audit.record("draft", agent=agent.name, phase="draft", attempt=0, text=draft_text)
        validation_feedback = ""
        last_errors: list[dict[str, object]] = []
        on_response = partial(self._usage_collector.record, agent_name=f"{agent.name}/structured_output") if self._usage_collector is not None else None

        conversion_instructions = f"""将以下分析素材整理为指定结构。不得添加素材中不存在的事实。
{CONCISE_ANALYSIS_OUTPUT_POLICY}
字段分工：decision.summary 只写一句话结论；evidence 保留选中事实的来源、原始URL、日期、关键数字与必要依据；reasoning/adopted_reasoning 只解释取舍、传导或新增调整，明确最强反证及撤回条件，不重抄 evidence。
事件、话题、驱动因素和 viewpoints 仅填写各自相关信息，不在多个字段复述同一事实；必要风险与字段专属口径限制仍须保留。不得把仅重复表述换成新事实。
保留素材中的股票身份、日期、货币、行情引用和原始数字，不改写来源URL、不近似替换数值；所有公共字段按原 schema 填写，必填字符串简短，无依据项按 schema 留空或 null，不编造补齐。
完整保留适用的 trade_intent、持仓假设、仓位、执行价与区间、目标、止损、执行步骤、风险控制和失效条件。只精简重复叙述，不删减这些安全与执行信息，不因证据条数丢弃关键反证。
预测仍仅覆盖下一实际交易日收盘；time_stop_days 是独立持仓管理期限，素材中的正整数（如3或5）不得强制改为1，纯退出或无仓位时仍为 null。
"""
        if issubclass(output_model, MarketAnalystOutput):
            conversion_instructions += (
                "期权字段仅使用被采纳的OI参考位；支撑/阻力位弃用时保留有效PCR，价位数组为空，不用技术价位补齐。"
                "非唯一Max Pain只保留区间解释，max_pain 为 null；保留到期日、PCR口径、OI更新时间和覆盖限制及弃用原因，未知覆盖同时在 interpretation 与 limitations 简短说明。\n"
            )
        if issubclass(output_model, SocialMediaAnalystOutput):
            conversion_instructions += (
                "独立社交采样缺失只在 data_quality 中集中说明一次，新闻情绪代理身份也归此处，不在其他字段重复；"
                "limitations 只写其他具体限制，没有其他限制时为空列表 []；合并素材须分开归属并保留其他限制原意。\n"
            )

        for attempt in range(1, self._max_validation_attempts + 1):
            content = f"{conversion_instructions}\n分析素材：\n{draft_text}\n{validation_feedback}"
            if audit is not None:
                audit.record("structured_request", agent=agent.name, phase="structured", attempt=attempt, text=content, schema=output_model.model_json_schema())
            try:
                try:
                    response = await self._formatter_model.generate_structured_output(
                        messages=[UserMsg(name="structured_output", content=content)],
                        structured_model=output_model,
                        on_response=on_response,
                    )
                except StructuredOutputError as exc:
                    cause = exc.__cause__
                    while cause is not None and not isinstance(cause, ValidationError):
                        cause = cause.__cause__
                    if cause is not None:
                        # The cause already chains back to exc; `from exc` would create a cycle.
                        raise cause from None
                    raise RuntimeError(str(exc)) from exc
                if audit is not None:
                    date = response.content.get("analysis_date") if isinstance(response.content, dict) else None
                    audit.record("structured_response", agent=agent.name, phase="structured", attempt=attempt, content=response.content, analysis_date=date)
                output = output_model.model_validate(response.content)
                validate_generated_output(output, reference_outputs)
                return output
            except ValidationError as exc:
                last_errors = exc.errors(
                    include_url=False,
                    include_context=False,
                    include_input=False,
                )
                lines = _format_validation_errors(last_errors)
                validation_feedback = "\n上次输出存在以下问题，请修正：\n" + "\n".join(lines)
            except GeneratedOutputPolicyError as exc:
                last_errors = exc.errors
                lines = _format_validation_errors(last_errors)
                validation_feedback = "\n上次输出未满足交易决策规则，请修正：\n" + "\n".join(lines)
            except (RuntimeError, ToolJSONDecodeError) as exc:
                last_errors = [{"loc": (), "msg": str(exc)}]
                lines = _format_validation_errors(last_errors)
                validation_feedback = "\n上次未生成符合 schema 的结构化对象，请仅返回指定结构。"

            if audit is not None:
                audit.record("structured_error", agent=agent.name, phase="structured", attempt=attempt, errors=last_errors)
            logger.warning(
                "[%s] structured validation attempt %d/%d failed: %s",
                agent.name,
                attempt,
                self._max_validation_attempts,
                redact("; ".join(lines)),
            )

        raise StructuredOutputValidationError(agent.name, last_errors)
