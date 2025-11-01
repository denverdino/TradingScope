# TradingScope

This project aims to create a Multi-Agents trading framework using [AgentScope](https://github.com/agentscope-ai/agentscope).

It is a learning project migrated from the original projects:

* https://github.com/hsliuping/TradingAgents-CN
* https://github.com/TauricResearch/TradingAgents


## Installation

```bash
make install
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

1. Set your OpenAI API key as an environment variable:
   ```bash
   export DASHSCOPE_API_KEY='your-api-key-here'
   export OPENAI_API_KEY='your-api-key-here'
   export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   export ALPHA_VANTAGE_API_KEY='your-api-key-here'
   ```

NOTE: This project can support DashScope Open API for Qwen model only at this time.

2. Run the example script:
   ```bash
   python -m tradingscope.main AAPL
   ```

3. Run the example script and send the report by email
   ```bash
   export EMAIL_FROM=abc@gmail.com
   export EMAIL_PASSWORD=xxxxxxx
   export SMTP_SSL_HOST=smtp.gmail.com
   export SMTP_SSL_PORT=465
   python -m tradingscope.main AAPL --email_to test@xyz.com
   ```

See [agents documentation](tradingscope/agents/README.md) for more details.

## Development

This project uses:
- `ruff` for linting
- `black` for formatting
- `pytest` for testing

## Data Storage

This project now uses file-based caching for storing stock data and analysis results. 