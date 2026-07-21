"""Tests for the shared AgentScope tracing lifecycle."""

from unittest.mock import MagicMock, patch

import pytest
from agentscope.middleware import TracingMiddleware

from tradingscope.agents.utils.tracing import (
    create_tracing_middlewares,
    setup_tracing,
    shutdown_tracing,
)


def test_tracing_is_disabled_when_environment_variable_is_absent(monkeypatch):
    monkeypatch.delenv("TRACING_ENABLED", raising=False)

    with (
        patch("tradingscope.agents.utils.tracing.TracerProvider") as provider_class,
        patch("tradingscope.agents.utils.tracing.OTLPSpanExporter") as exporter_class,
    ):
        provider = setup_tracing("tradingscope-main")

    assert provider is None
    provider_class.assert_not_called()
    exporter_class.assert_not_called()
    assert create_tracing_middlewares() == []
    shutdown_tracing(None)


def test_setup_tracing_configures_and_registers_provider(monkeypatch):
    monkeypatch.setenv("TRACING_ENABLED", "1")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "http://collector.example/v1/traces",
    )
    provider = MagicMock()
    resource = MagicMock()
    exporter = MagicMock()
    processor = MagicMock()

    with (
        patch(
            "tradingscope.agents.utils.tracing.TracerProvider",
            return_value=provider,
        ),
        patch("tradingscope.agents.utils.tracing.Resource") as resource_class,
        patch(
            "tradingscope.agents.utils.tracing.OTLPSpanExporter",
            return_value=exporter,
        ) as exporter_class,
        patch(
            "tradingscope.agents.utils.tracing.BatchSpanProcessor",
            return_value=processor,
        ),
        patch(
            "tradingscope.agents.utils.tracing.trace.set_tracer_provider",
        ) as set_tracer_provider,
        patch(
            "tradingscope.agents.utils.tracing.trace.get_tracer_provider",
            return_value=provider,
        ),
    ):
        resource_class.create.return_value = resource

        result = setup_tracing("tradingscope-main")

    assert result is provider
    resource_class.create.assert_called_once_with(
        {"service.name": "tradingscope-main"},
    )
    exporter_class.assert_called_once_with(
        endpoint="http://collector.example/v1/traces",
    )
    provider.add_span_processor.assert_called_once_with(processor)
    set_tracer_provider.assert_called_once_with(provider)


def test_setup_tracing_uses_default_endpoint(monkeypatch):
    monkeypatch.setenv("TRACING_ENABLED", "1")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    provider = MagicMock()

    with (
        patch(
            "tradingscope.agents.utils.tracing.TracerProvider",
            return_value=provider,
        ),
        patch("tradingscope.agents.utils.tracing.Resource"),
        patch(
            "tradingscope.agents.utils.tracing.OTLPSpanExporter",
        ) as exporter_class,
        patch("tradingscope.agents.utils.tracing.BatchSpanProcessor"),
        patch("tradingscope.agents.utils.tracing.trace.set_tracer_provider"),
        patch(
            "tradingscope.agents.utils.tracing.trace.get_tracer_provider",
            return_value=provider,
        ),
    ):
        setup_tracing("tradingscope-main")

    exporter_class.assert_called_once_with(
        endpoint="http://localhost:3000/v1/traces",
    )


def test_setup_tracing_shuts_down_rejected_provider_and_raises(monkeypatch):
    monkeypatch.setenv("TRACING_ENABLED", "1")
    provider = MagicMock()
    active_provider = MagicMock()

    with (
        patch(
            "tradingscope.agents.utils.tracing.TracerProvider",
            return_value=provider,
        ),
        patch("tradingscope.agents.utils.tracing.Resource"),
        patch("tradingscope.agents.utils.tracing.OTLPSpanExporter"),
        patch("tradingscope.agents.utils.tracing.BatchSpanProcessor"),
        patch("tradingscope.agents.utils.tracing.trace.set_tracer_provider"),
        patch(
            "tradingscope.agents.utils.tracing.trace.get_tracer_provider",
            return_value=active_provider,
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="OpenTelemetry tracer provider registration was rejected",
        ):
            setup_tracing("tradingscope-main")

    provider.shutdown.assert_called_once_with()


def test_create_tracing_middlewares_returns_tracing_middleware_when_enabled(
    monkeypatch,
):
    monkeypatch.setenv("TRACING_ENABLED", "1")

    middlewares = create_tracing_middlewares()

    assert len(middlewares) == 1
    assert isinstance(middlewares[0], TracingMiddleware)


def test_shutdown_tracing_shuts_down_provider():
    provider = MagicMock()

    shutdown_tracing(provider)

    provider.shutdown.assert_called_once_with()
