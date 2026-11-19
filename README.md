# TradingScope

A multi-agent trading analysis framework built on [AgentScope 2.0](https://github.com/agentscope-ai/agentscope) and DashScope Qwen models.

This learning project is inspired by:

- [TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN)
- [TradingAgents](https://github.com/TauricResearch/TradingAgents)


## Installation

Python 3.11 or later and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev
```

## Usage

```bash
make lint          # Run Ruff lint checks
make lint-fix      # Run Ruff and apply safe fixes
make format        # Format with Ruff
make format-check  # Check formatting
make imports       # Sort imports with Ruff
make test          # Run all pytest tests
make test-cov      # Run tests with coverage
make evaluate      # Evaluate the seven scheduled tickers; override with TICKERS=AAPL,MSFT
```

### Running the Analysis

1. Set your API keys as environment variables:

   ```bash
   export DASHSCOPE_API_KEY='your-api-key-here'

   # Optional
   export ALPHA_VANTAGE_API_KEY='your-api-key-here'
   export PERPLEXITY_API_KEY='your-api-key-here'

   # Optional: OSS report storage
   export OSS_ACCESS_KEY_ID='your-access-key-id'
   export OSS_ACCESS_KEY_SECRET='your-access-key-secret'
   export OSS_REGION='cn-hangzhou'
   export OSS_BUCKET='your-bucket-name'
   ```

   TradingScope uses the DashScope SDK for Qwen models. No OpenAI API key is required. Alpha Vantage, Perplexity, and OSS credentials are needed only when their corresponding integrations are used.

2. Run the analysis:

   ```bash
   uv run python -m tradingscope.main AAPL
   ```

   Use `--output markdown`, `--output json`, or `--output both` (the default) to control local output. The installed `tradingscope AAPL` command is equivalent.

3. Run with email report delivery:

   ```bash
   export EMAIL_FROM=abc@gmail.com
   export EMAIL_PASSWORD=xxxxxxx
   export SMTP_SSL_HOST=smtp.gmail.com
   export SMTP_SSL_PORT=465
   uv run python -m tradingscope.main AAPL --email-to test@xyz.com
   ```

   `--email-to` accepts a comma-separated recipient list. Email is sent only when Markdown output is enabled. `SMTP_SSL_HOST` and `SMTP_SSL_PORT` optionally override the SMTP endpoint.

### Post-market Evaluation

Each analysis forecasts only the next actual trading session through its close: the first session on or after the analysis date, skipping weekends and holidays. Evaluation deterministically scores all seven stages at that single-session horizon, using the previous completed close as the return baseline, including overnight gaps. It does not call an LLM:

```bash
# Evaluate pending analyses from the previous 14 calendar dates for selected tickers
uv run python -m tradingscope.evaluate --tickers AAPL,MSFT

# Evaluate a specific analysis date
uv run python -m tradingscope.evaluate --tickers AAPL --date 2026-07-15

# Compute and print only: no record writes and no email
uv run python -m tradingscope.evaluate --tickers AAPL --dry-run

# Send the generated evaluation summary
uv run python -m tradingscope.evaluate --tickers AAPL --email-to team@example.com
```

The installed `tradingscope-evaluate` command provides the same interface. `make evaluate` defaults to MSFT, AMZN, INTC, NVDA, GOOGL, BABA and TSM; `make evaluate TICKERS=AAPL,MSFT` selects different tickers.

Evaluation requires OSS read access and market-data access, but no DashScope credentials. Evaluation accepts only schema-3.1 completion manifests and their immutable full-report bundles. The current analysis output schema remains 3.0. Older manifests, renamed historical fields, incomplete bundles and missing trade intents are rejected, not upgraded or inferred. Each run's single-session evaluation is written once as `tradingscope-evaluations/<analysis_date>/<ticker>/<run_id>/h1.json`, including copies of the source stage outputs. With OSS configured, records are stored there; without OSS, both analysis sources and evaluation records are read/written under `--results-dir`, using either a specified analysis date or the recent pending window. Existing records prevent duplicate market-data requests and writes. Schema-3.1 manifests carry a run ID, code version, model names, last completed session and forecast start/completion timestamps. Completed reads use the immutable `tradingscope/<analysis_date>/<ticker>/runs/<run_id>/full_report.json` bundle, never the mutable per-node previews, so overlapping processes cannot mix stage outputs. A same-date rerun gets separate evaluations; only the latest complete run is evaluated and counted toward Track Record. Superseded records remain archived and are not re-evaluated. Archived 3- and 5-session evaluations are excluded from every Track Record statistic, not converted into single-session results. Old data is not backfilled or included in Track Record; `--backfill` has been removed. Existing historical files are not deleted.

Direction Skill uses excess return over the ticker's sector ETF (SPY if unmapped), with an ATR band for inconclusive moves. Sessions with benchmark-relative daily moves or opening gaps above three daily ATR percentages are labelled as shocks. Unscheduled or unclassified shocks are excluded from Direction Skill, but remain in Risk Control. Historical earnings release timestamps identify the reaction session: premarket releases apply to that session, post-close releases to the next session. Intraday or missing release timing is treated as unknown, not as an excuse. Historical calendars alone do not prove that the schedule was known at analysis time. All stages receive direction scores; only Trader and Portfolio Manager receive execution simulations. Daily-bar evaluation rejects forecasts made for an already-completed session or completed after their first future session opened; both forecast timestamps are required. Intraday simulations retain stop-before-target precedence when daily OHLC cannot establish event order. An open position not exited during the forecast session is valued at its close; `horizon` is a valuation, not forced liquidation. The same distinction applies to residual positions after a reduction. `time_stop_days` remains an independent holding deadline and may exceed one session; pure exits and flat plans keep it empty. Holding may continue, but a later-session forecast requires fresh analysis. Return statistics are separated by trade intent and return basis, and residual-position stop events and simulation limitations remain visible. Track Record uses only single-session evaluations and separates ticker/global groups, roles, confidence buckets and analyst/PM agreement; ratios are hidden below ten samples. Thresholds and the reset date live in `DEFAULT_CONFIG["evaluation"]`. No lessons are generated or injected yet.

The scheduled workflow runs once at 08:00 Beijing time, after the US close. The analysis CLI skips a ticker when a recent complete run already covered its last completed session, avoiding weekend/holiday duplicates. Use `--force`, or the manual workflow's `force` input, to rerun deliberately.

### OpenTelemetry Tracing

Tracing is opt-in. Set a non-empty `TRACING_ENABLED` value to attach AgentScope's `TracingMiddleware` and export spans through OTLP/HTTP:

```bash
export TRACING_ENABLED=1
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT='http://localhost:3000/v1/traces'

uv run python -m tradingscope.main AAPL
uv run python -m tradingscope.evaluate --tickers AAPL
```

The endpoint defaults to `http://localhost:3000/v1/traces`. Analysis traces use service name `tradingscope-main`; evaluation traces use `tradingscope-evaluation`. Both CLIs own the tracer-provider lifecycle and shut it down on success or failure so queued spans are flushed. Leave `TRACING_ENABLED` unset or empty to disable tracing.

## Architecture

The system uses a 7-stage pipeline with multiple AI agents:

1. **Analysis** — Market, Fundamentals, News, and Social Media analysts run concurrently
2. **Research Debate** — Bull/Bear researchers debate investment thesis
3. **Research Management** — Research Manager synthesizes debate into recommendation
4. **Trading Decision** — Trader agent determines entry/stop-loss/target prices
5. **Risk Debate** — Aggressive/Conservative/Neutral risk analysts debate
6. **Risk Management** — Portfolio Manager makes final risk-adjusted decision
7. **Report Generation** — Markdown report + structured JSON output

The four analyst stages use `asyncio.gather`. All seven JSON-producing nodes are validated against strict Pydantic schema v3 models before downstream stages continue. Bull/Bear and risk debates exchange messages through their orchestrators and `agent.observe()`.

### AgentScope 2.0

This project uses AgentScope 2.0 (`agentscope>=2.0.6,<2.1`) with:

- `Agent` class with `ReActConfig` for tool-calling agents
- `DashScopeResponseModel` based on `OpenAIResponseModel`, with `OpenAICredential` for Alibaba Cloud's OpenAI-compatible Responses API and server-side Code Interpreter enabled by default
- `Toolkit(tools=[FunctionTool(...)])` for tool registration
- `agent.observe()` for multi-agent message broadcasting (replaces MsgHub)
- `ToolChunk` return type for all tool functions
- `TracingMiddleware` with an OpenTelemetry OTLP exporter for optional tracing

## Examples

Example scripts are in the `examples/` directory:

```bash
# Run a single analyst
uv run python -m examples.market_analyst
uv run python -m examples.fundamentals_analyst
uv run python -m examples.news_analyst
uv run python -m examples.social_media_analyst

# Run the research debate
uv run python -m examples.research_debate

# Run the risk management debate
uv run python -m examples.risk_management_debate

# Run the full workflow
uv run python -m examples.workflow

# Review saved analysis records
uv run python -m examples.memories --review AAPL
```

## Development

This project uses:

- `uv` for package management
- `ruff` for linting and formatting
- `pytest` for testing

## Data Storage

Stock data is cached under `tradingscope/dataflows/data_cache/`. Local analysis outputs are written under `results/data/<date>/<ticker>/` by default; set `TRADINGAGENTS_RESULTS_DIR` to change the root. Each validated workflow node is checkpointed there as JSON and Markdown when it finishes. Depending on `--output`, the CLI additionally writes an HTML report and/or a combined JSON result.

### OSS Report Storage

When all OSS environment variables are configured, the workflow uploads Markdown and JSON artifacts to Alibaba Cloud OSS at:

```
tradingscope/<date>/<ticker>/<agent_name>.{md,json}
```

Artifacts include `market_analyst`, `fundamentals_analyst`, `news_analyst`, `social_media_analyst`, `research_manager`, `trader`, `portfolio_manager`, and `full_report`. Each node's JSON and Markdown artifacts are uploaded as soon as that node finishes. Starting another run for the same ticker/date first invalidates its previous manifest; same-key runs within one process are serialized. A schema-v3 `manifest.json` is written last only after all required artifacts succeed; evaluation uses it as the completion marker. If OSS is not configured, the workflow still checkpoints every node locally and skips OSS persistence.
