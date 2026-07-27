"""Strict two-phase execution for typed agent outputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Generic, TypeVar

from agentscope.message import Msg, UserMsg
from pydantic import ValidationError

from tradingscope.agents.output import AgentOutputBase

from .agent_utils import call_agent_with_retry
from .decision_validation import GeneratedOutputPolicyError, validate_generated_output

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

    def __init__(self, formatter_model, max_validation_attempts: int = 3) -> None:
        if max_validation_attempts < 1:
            raise ValueError("max_validation_attempts must be at least 1")
        self._formatter_model = formatter_model
        self._max_validation_attempts = max_validation_attempts

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
        validation_feedback = ""
        last_errors: list[dict[str, object]] = []

        for attempt in range(1, self._max_validation_attempts + 1):
            content = f"将以下分析素材整理为指定结构。不得添加素材中不存在的事实。\n\n分析素材：\n{draft_text}\n{validation_feedback}"
            try:
                response = await self._formatter_model.generate_structured_output(
                    messages=[UserMsg(name="structured_output", content=content)],
                    structured_model=output_model,
                )
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
            except RuntimeError as exc:
                last_errors = [{"loc": (), "msg": str(exc)}]
                lines = _format_validation_errors(last_errors)
                validation_feedback = "\n上次未生成符合 schema 的结构化对象，请仅返回指定结构。"

            logger.warning(
                "[%s] structured validation attempt %d/%d failed: %s",
                agent.name,
                attempt,
                self._max_validation_attempts,
                "; ".join(lines),
            )

        raise StructuredOutputValidationError(agent.name, last_errors)
