# TradingScope Agent Guide

## Project Overview

TradingScope is a multi-agent trading analysis system built on [AgentScope 2.0](https://github.com/agentscope-ai/agentscope) and DashScope Qwen models. It is inspired by [TradingAgents](https://github.com/TauricResearch/TradingAgents).

Use Python 3.11 or later. Manage dependencies and commands with `uv`; do not introduce an OpenAI API dependency.

## Commands

```bash
uv sync --extra dev       # Install dependencies
make install              # Install dependencies with uv
make lint                 # Run Ruff lint checks
make lint-fix             # Run Ruff and apply safe fixes
make format               # Format with Ruff
make format-check         # Check formatting
make imports              # Sort imports with Ruff
make test                 # Run all pytest tests
make test-cov             # Run tests with coverage
make evaluate             # Run post-market evaluation
make clean                # Remove build and test artifacts
```

Targeted test commands:

```bash
uv run pytest tests/test_foo.py
uv run pytest tests/test_foo.py::TestClass::test_method -v
uv run pytest -m "not slow and not integration"
```

Run the application with `source .env && uv run python -m tradingscope.main AAPL` so live model credentials are loaded in the same shell.

Run evaluation with `uv run python -m tradingscope.evaluate --tickers AAPL,MSFT`. The `--tickers` argument is required; `make evaluate` does not supply it.

## Environment Variables

- `DASHSCOPE_API_KEY` — required for Qwen model calls.
- `ALPHA_VANTAGE_API_KEY` — required only when using Alpha Vantage data.
- `PERPLEXITY_API_KEY` — required only when using the Perplexity data source.
- `OSS_ACCESS_KEY_ID`, `OSS_ACCESS_KEY_SECRET`, `OSS_REGION`, `OSS_BUCKET` — optional Alibaba Cloud OSS report storage.
- `EMAIL_FROM`, `EMAIL_PASSWORD` — required only when email delivery is requested. SMTP host and port may be configured with `SMTP_SSL_HOST` and `SMTP_SSL_PORT`.
- `TRADINGAGENTS_RESULTS_DIR` — optional local output and evaluation-state root; defaults to `./results`.
- `TRACING_ENABLED` — enables AgentScope/OpenTelemetry tracing when set to any non-empty value.
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` — optional OTLP/HTTP traces endpoint; defaults to `http://localhost:3000/v1/traces`.
- `MEMORY_DEBUG` — enables extra evaluation-related debug logging when set.

Never commit credentials, tokens, generated reports, or local environment files.
For any live application or integration-test command, source `~/daily/llm_env.sh` in the same shell. Do not print, inspect, or persist the loaded secrets.

## Architecture and Code Map

The workflow in `tradingscope/agents/workflow.py` has seven stages:

1. **Analysis** — Market, Fundamentals, News, and Social Media analysts run concurrently with `asyncio.gather` and produce Markdown plus structured output.
2. **Research Debate** — Bull and Bear researchers exchange statements, rebuttals, and convergence messages through `ResearchDebateOrchestrator` and `agent.observe()`.
3. **Research Management** — The Research Manager synthesizes the debate into an investment recommendation.
4. **Trading Decision** — The Trader produces buy, sell, or hold guidance with entry, stop-loss, and target prices.
5. **Risk Debate** — Aggressive, Conservative, and Neutral agents debate the proposed trade.
6. **Risk Management** — The Portfolio Manager makes the final risk-adjusted decision.
7. **Report Generation** — The workflow returns a Markdown report and structured `AnalysisResult`, with optional OSS upload or email delivery.

Key locations:

- `tradingscope/main.py` — analysis CLI entry point.
- `tradingscope/evaluate.py` — post-market evaluation CLI entry point.
- `tradingscope/default_config.py` — model, debate, directory, and default vendor configuration.
- `tradingscope/agents/workflow.py` — end-to-end pipeline orchestration.
- `tradingscope/agents/utils/context.py` — shared `AgentContext`, model initialization, and report state.
- `tradingscope/agents/output.py` — Pydantic schemas for stage outputs and `AnalysisResult`.
- `tradingscope/agents/renderers.py` — Markdown renderers for validated structured outputs.
- `tradingscope/agents/utils/structured_output.py` — shared structured-output execution and validation.
- `tradingscope/agents/utils/dashscope_response_model.py` — DashScope Responses API adapter, including structured-output schema normalization and thinking-mode tool selection.
- `tradingscope/agents/evaluation/evaluator.py` — market-result scoring and LLM-generated evaluations/lessons.
- `tradingscope/agents/evaluation/oss_store.py` — manifest-gated OSS ingestion and local evaluated-state tracking.
- `tradingscope/agents/utils/tracing.py` — opt-in OpenTelemetry provider and AgentScope middleware lifecycle.
- `tradingscope/dataflows/interface.py` — `VENDOR_METHODS` and `route_to_vendor()` fallback routing.
- `tradingscope/dataflows/config.py` — runtime dataflow configuration.
- `tradingscope/utils/oss_structured_output_uploader.py` — schema-v2 JSON/Markdown persistence and completion manifest.
- `tradingscope/utils/oss_structured_output_reader.py` — strict completion-manifest validation and structured OSS reads.
- `tradingscope/agents/utils/*_tools.py` — AgentScope tool wrappers returning `ToolChunk` values through `@agentscope_tool`.
- `tests/` — pytest suite; mirror the source area when adding focused tests.

Vendor selection follows this precedence: tool-level `tool_vendors`, category-level `data_vendors`, then the fallback order implemented by `route_to_vendor()`. Vendor preferences may be comma-separated.

The configured models are:

- `deep_think_llm`: `qwen3.8-max` for analysis and debate with thinking enabled.
- `quick_think_llm`: `qwen3.7-plus` for fast tasks.
- `non_thinking_model`: the deep model with thinking disabled for debate agents where extra reasoning is unnecessary.

### Structured Output and Persistence

All seven JSON-producing workflow nodes run through `StructuredAgentRunner`; schema validation failures stop downstream processing. The workflow returns one schema-v2 `AnalysisResult`. Local CLI output goes to `<results_dir>/data/<date>/<ticker>/` as HTML and/or JSON according to `--output`.

`DashScopeResponseModel` must inline local Pydantic `$ref` definitions before sending structured-output tools because DashScope may otherwise serialize nested objects as JSON strings. Preserve Pydantic model validation when normalizing schemas. Thinking-enabled requests use `tool_choice="auto"`; non-thinking structured-output requests use `tool_choice="required"`.

OSS persistence is optional for analysis. When configured, `persist_analysis_result()` uploads Markdown and JSON for every node and `full_report`, then writes `manifest.json` last. Treat the manifest as the atomic completion marker: consumers must use `fetch_completed_v2_output()` or `async_fetch_completed_v2_output()` rather than reading an artifact without validating the manifest.

### Post-market Evaluation

The evaluation CLI requires `--tickers` and optionally accepts `--date`, `--results-dir`, `--dry-run`, and `--email-to`. `OSSAnalysisStore` reads only manifest-completed schema-v2 `portfolio_manager.json` objects from OSS and converts validated `PortfolioManagerOutput` values into `AnalysisRecord` objects.

`AnalysisEvaluator` fetches market data through `route_to_vendor("get_stock_data", ...)`, compares the analysis-date close with the immediately preceding trading-day close, checks direction and stop-loss behavior, and uses `Agent` plus `non_thinking_model` to generate a Chinese evaluation and lesson. The CLI currently passes `memory_manager=None`, so lessons are returned/logged but not written to shared memory. Unless `--dry-run` is used, completed keys are recorded in `<results_dir>/oss_evaluated.json`; optional evaluation email uses the same email environment variables as analysis delivery.

### Tracing

Tracing is disabled unless `TRACING_ENABLED` is non-empty. `setup_tracing()` creates an OpenTelemetry `TracerProvider`, service resource, OTLP/HTTP exporter, and `BatchSpanProcessor`; `AgentContext` supplies `TracingMiddleware` to AgentScope agents. The analysis and evaluation entry points register service names `tradingscope-main` and `tradingscope-evaluation`, respectively, and always call `shutdown_tracing()` in `finally` blocks to flush and close the provider.

Keep setup at the CLI boundary. Do not initialize another global provider inside agents or workflows. Tests that exercise both CLIs in one process should mock or isolate the provider lifecycle because OpenTelemetry accepts only one global provider registration.

## Working Rules

### Before Editing

- Inspect the relevant implementation, tests, configuration, and current Git diff before proposing changes.
- State assumptions that affect behavior. If requirements have materially different interpretations, present the trade-off and ask instead of choosing silently.
- Prefer the simpler solution when it fully satisfies the request. Push back on unnecessary scope.
- For multi-step work, define a short plan with a verification result for each step.

### While Editing

- Make the smallest change that solves the requested problem. Do not add speculative features, flexibility, abstractions, or handling for impossible cases.
- Follow SOLID, DRY, and YAGNI pragmatically. Extract shared code only when there is real reuse; do not create abstractions for a single use.
- Match the surrounding style and architecture. Do not refactor, reformat, or clean up unrelated code.
- Preserve all unrelated user changes in a dirty worktree. Mention unrelated dead code rather than deleting it.
- Remove imports, variables, functions, and files made obsolete by your own change.
- Every changed line must trace to the request or to verification required by the request.

### Testing and Verification

- For a bug fix, first add or identify a test that reproduces the failure, then make it pass.
- For new behavior, add focused tests covering the requested success and failure cases.
- Run the narrowest relevant tests during iteration, then expand checks in proportion to the affected scope.
- After Python changes, run `make lint-fix`, relevant pytest tests, and `make format-check`. Run the full suite when changes cross modules or affect the workflow.
- For documentation-only changes, inspect the rendered structure, verify commands and paths against the repository, and review the final diff; the application test suite is not required.
- Do not claim success without fresh command output. Report skipped checks and the reason.

### Completion

Summarize:

1. What changed and why.
2. What was verified, including exact commands.
3. Any remaining risk, assumption, or follow-up.

- Never commit .env file
- Never commit AI-generated design, specification, planning, or implementation-plan documents. Keep such artifacts outside the repository or leave them untracked for local use only.
- Before merging any development branch into a trunk branch such as `main` or `master`, squash all branch commits into a single commit. Do not merge a multi-commit branch history directly into the trunk.
- After repository changes are complete and verified, automatically create the squashed local commit. This is standing authorization for local commits and local merges; do not ask for confirmation each time.

Do not push, open a pull request, upload reports, or send email unless the user explicitly requests it.
