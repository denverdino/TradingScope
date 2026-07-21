"""Shared lifecycle helpers for opt-in AgentScope tracing."""

import os

from agentscope.middleware import TracingMiddleware
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

DEFAULT_TRACING_ENDPOINT = "http://localhost:3000/v1/traces"


def tracing_enabled() -> bool:
    return bool(os.getenv("TRACING_ENABLED"))


def create_tracing_middlewares() -> list[TracingMiddleware]:
    return [TracingMiddleware()] if tracing_enabled() else []


def setup_tracing(service_name: str) -> TracerProvider | None:
    if not tracing_enabled():
        return None
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name}),
    )
    exporter = OTLPSpanExporter(
        endpoint=os.getenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            DEFAULT_TRACING_ENDPOINT,
        ),
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    if trace.get_tracer_provider() is not provider:
        provider.shutdown()
        raise RuntimeError(
            "OpenTelemetry tracer provider registration was rejected; a global provider is already active",
        )
    return provider


def shutdown_tracing(provider: TracerProvider | None) -> None:
    if provider is not None:
        provider.shutdown()
