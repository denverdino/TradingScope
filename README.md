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
make evaluate      # Run post-market evaluation (requires --tickers)
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

Evaluation compares a completed portfolio decision with market closes, then asks the non-thinking Qwen model to produce a concise Chinese evaluation and lesson:

```bash
# Evaluate the latest eligible US trading day for selected tickers
uv run python -m tradingscope.evaluate --tickers AAPL,MSFT

# Evaluate a specific analysis date
uv run python -m tradingscope.evaluate --tickers AAPL --date 2026-07-15

# Preview without recording the report as evaluated
uv run python -m tradingscope.evaluate --tickers AAPL --dry-run

# Send the generated evaluation summary
uv run python -m tradingscope.evaluate --tickers AAPL --email-to team@example.com
```

The installed `tradingscope-evaluate` command provides the same interface. `make evaluate` invokes the module directly, so pass arguments with `uv run python -m tradingscope.evaluate ...` when ticker selection is required.

Evaluation requires DashScope, access to the selected market-data vendor, and configured OSS storage. It reads only completed schema-v2 results from `tradingscope/<date>/<ticker>/portfolio_manager.json`, gated by a valid `manifest.json`. It calculates the return from the previous trading-day close to the analysis-date close, checks direction and stop-loss behavior, and logs the generated result. Successfully processed records are tracked locally in `<results_dir>/oss_evaluated.json`; set `TRADINGAGENTS_RESULTS_DIR` or use `--results-dir` to change that location. `--dry-run` skips this tracking update.

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

The four analyst stages use `asyncio.gather`. All seven JSON-producing nodes are validated against strict Pydantic schema v2 models before downstream stages continue. Bull/Bear and risk debates exchange messages through their orchestrators and `agent.observe()`.

### AgentScope 2.0

This project uses AgentScope 2.0 (`agentscope>=2.0.4,<2.1`) with:

- `Agent` class with `ReActConfig` for tool-calling agents
- `DashScopeChatModel` with `DashScopeCredential` for model configuration
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

Stock data is cached under `tradingscope/dataflows/data_cache/`. Local analysis outputs are written under `results/data/<date>/<ticker>/` by default; set `TRADINGAGENTS_RESULTS_DIR` to change the root. Depending on `--output`, the CLI writes an HTML report, per-agent JSON files, and a combined JSON result.

### OSS Report Storage

When all OSS environment variables are configured, the workflow uploads Markdown and JSON artifacts to Alibaba Cloud OSS at:

```
tradingscope/<date>/<ticker>/<agent_name>.{md,json}
```

Artifacts include `market_analyst`, `fundamentals_analyst`, `news_analyst`, `social_media_analyst`, `research_manager`, `trader`, `portfolio_manager`, and `full_report`. A schema-v2 `manifest.json` is written last only after all required artifacts succeed; evaluation uses it as the completion marker. If OSS is not configured, analysis still writes local output and skips OSS persistence.
