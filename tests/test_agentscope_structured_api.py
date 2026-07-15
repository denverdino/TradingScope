"""Contract tests for the AgentScope structured-output API."""

from __future__ import annotations

import inspect
from importlib.metadata import version

from agentscope.model import DashScopeChatModel
from packaging.version import Version


def test_agentscope_exposes_structured_output_api() -> None:
    assert Version(version("agentscope")) >= Version("2.0.4")

    method = DashScopeChatModel.generate_structured_output
    parameters = inspect.signature(method).parameters

    assert inspect.iscoroutinefunction(method)
    assert "messages" in parameters
    assert "structured_model" in parameters
