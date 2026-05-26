# TradingScope

A Multi-Agent trading analysis framework built on [AgentScope 2.0](https://github.com/agentscope-ai/agentscope).

It is a learning project migrated from the original projects:

* https://github.com/hsliuping/TradingAgents-CN
* https://github.com/TauricResearch/TradingAgents


## Installation

```bash
uv sync --extra dev
```

## Usage

```bash
make lint        # Run linting
make format      # Format code
make imports     # Order imports
make test        # Run tests
make test-cov    # Run tests with coverage
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

   NOTE: This project uses the DashScope SDK for Qwen models as its LLM provider. No OpenAI API key is required.

2. Run the analysis:
   ```bash
   uv run python -m tradingscope.main AAPL
   ```

3. Run with email report delivery:
   ```bash
   export EMAIL_FROM=abc@gmail.com
   export EMAIL_PASSWORD=xxxxxxx
   export SMTP_SSL_HOST=smtp.gmail.com
   export SMTP_SSL_PORT=465
   uv run python -m tradingscope.main AAPL --email-to test@xyz.com
   ```

## Architecture

The system uses a 7-stage pipeline with multiple AI agents:

1. **Analysis** — Market, Fundamentals, News, and Social Media analysts run concurrently
2. **Research Debate** — Bull/Bear researchers debate investment thesis
3. **Research Management** — Research Manager synthesizes debate into recommendation
4. **Trading Decision** — Trader agent determines entry/stop-loss/target prices
5. **Risk Debate** — Aggressive/Conservative/Neutral risk analysts debate
6. **Risk Management** — Portfolio Manager makes final risk-adjusted decision
7. **Report Generation** — Markdown report + structured JSON output

### AgentScope 2.0

This project uses AgentScope 2.0 (`agentscope>=2.0.0`) with:
- `Agent` class with `ReActConfig` for tool-calling agents
- `DashScopeChatModel` with `DashScopeCredential` for model configuration
- `Toolkit(tools=[FunctionTool(...)])` for tool registration
- `agent.observe()` for multi-agent message broadcasting (replaces MsgHub)
- `ToolChunk` return type for all tool functions

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

This project uses file-based caching for storing stock data and analysis results.

### OSS Report Storage

When OSS environment variables are configured, agent-generated reports are automatically uploaded to Alibaba Cloud OSS at:

```
tradingscope/<date>/<ticker>/<agent_name>.md
```

Reports uploaded include: `market_analyst`, `fundamentals_analyst`, `news_analyst`, `social_media_analyst`, `research_manager`, `trader`, `portfolio_manager`, and `full_report`. If OSS is not configured, the workflow runs normally and skips uploads.
