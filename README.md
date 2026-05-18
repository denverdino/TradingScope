# TradingScope

This project aims to create a Multi-Agents trading framework using [AgentScope](https://github.com/agentscope-ai/agentscope).

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


### Running the Fundamentals Analyst Agent Example

To run the fundamentals analyst agent example:

1. Set your API keys as environment variables:
   ```bash
   export DASHSCOPE_API_KEY='your-api-key-here'

   # Optional
   export ALPHA_VANTAGE_API_KEY='your-api-key-here'
   export PERPLEXITY_API_KEY='your-api-key-here'
   ```

   NOTE: This project uses the DashScope SDK for Qwen models as its LLM provider. No OpenAI API key is required.

   # Optional: OSS report storage
   export OSS_ACCESS_KEY_ID='your-access-key-id'
   export OSS_ACCESS_KEY_SECRET='your-access-key-secret'
   export OSS_REGION='cn-hangzhou'
   export OSS_BUCKET='your-bucket-name'
   ```

NOTE: This project can support DashScope Open API for Qwen model only at this time.

2. Run the example script:
   ```bash
   uv run python -m tradingscope.main AAPL
   ```

3. Run the example script and send the report by email
   ```bash
   export EMAIL_FROM=abc@gmail.com
   export EMAIL_PASSWORD=xxxxxxx
   export SMTP_SSL_HOST=smtp.gmail.com
   export SMTP_SSL_PORT=465
   uv run python -m tradingscope.main AAPL --email_to test@xyz.com
   ```

See [agents documentation](tradingscope/agents/README.md) for more details.

## Development

This project uses:
- `uv` for package management
- `ruff` for linting and formatting
- `pytest` for testing

## Data Storage

This project now uses file-based caching for storing stock data and analysis results.

### OSS Report Storage

When OSS environment variables are configured, agent-generated reports are automatically uploaded to Alibaba Cloud OSS at:

```
tradingscope/<date>/<ticker>/<agent_name>.md
```

Reports uploaded include: `market_analyst`, `fundamentals_analyst`, `news_analyst`, `social_media_analyst`, `research_manager`, `trader`, `portfolio_manager`, and `full_report`. If OSS is not configured, the workflow runs normally and skips uploads. 