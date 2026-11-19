"""Strict two-phase execution for typed agent outputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Generic, TypeVar

from agentscope.exception import StructuredOutputError, ToolJSONDecodeError
from agentscope.message import Msg, UserMsg
from pydantic import ValidationError

from tradingscope.agents.output import AgentOutputBase

from .agent_utils import call_agent_with_retry
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

        for attempt in range(1, self._max_validation_attempts + 1):
            content = f"将以下分析素材整理为指定结构。不得添加素材中不存在的事实。\n\n分析素材：\n{draft_text}\n{validation_feedback}"
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
