# TradingScope Project Context

## Project Overview

TradingScope is a multi-agent trading analysis system built on [AgentScope](https://github.com/agentscope-ai/agentscope) using DashScope (Qwen models) as the LLM provider.

References:

* https://github.com/TauricResearch/TradingAgents

## Build & Development Commands

```bash
uv sync --extra dev   # Install dependencies with uv
make install          # Install dependencies with uv
make lint             # Lint with ruff
make lint-fix         # Lint and auto-fix violations
make format           # Format code with ruff
make format-check     # Check formatting without changes
make imports          # Order imports with ruff
make test             # Run all tests (pytest)
make test-cov         # Run tests with coverage report
make evaluate         # Run post-market evaluation CLI
make clean            # Clean build artifacts
```

Run a single test file: `uv run pytest tests/test_foo.py`
Run a single test: `uv run pytest tests/test_foo.py::TestClass::test_method -v`
Skip slow/integration tests: `uv run pytest -m "not slow and not integration"`

## Required Environment Variables

- `DASHSCOPE_API_KEY` — required for Qwen LLM models (DashScope SDK, no OpenAI key needed)
- `ALPHA_VANTAGE_API_KEY` — optional for Alpha Vantage data APIs
- `PERPLEXITY_API_KEY` — optional, for Perplexity data source
- OSS variables (`OSS_ACCESS_KEY_ID`, `OSS_ACCESS_KEY_SECRET`, `OSS_REGION`, `OSS_BUCKET`) — optional, for report upload to Alibaba Cloud OSS

## Architecture

The pipeline runs 7 stages:

1. **Analysis** — Four analyst agents (Market, Fundamentals, News, Social Media) run concurrently via `asyncio.gather`. Each uses AgentScope's `ReActAgent` with tool access to dataflows, producing Markdown reports + Pydantic structured outputs.
2. **Research Debate** — Bull/Bear researchers debate via `ResearchDebateOrchestrator` using `MsgHub` (progressive rounds: statement → rebuttal → convergence).
3. **Research Management** — Research Manager synthesizes debate into an investment recommendation.
4. **Trading Decision** — Trader agent makes buy/sell/hold decision with entry/stop-loss/target prices.
5. **Risk Debate** — Aggressive/Conservative/Neutral debators debate via `RiskDebateOrchestrator`.
6. **Risk Management** — Portfolio Manager evaluates risk debate for final risk-adjusted decision.
7. **Report Generation** — Markdown report + structured JSON (`AnalysisResult`) produced; optionally uploaded to OSS or emailed.

### Key Design Patterns

**Vendor routing** (`dataflows/interface.py`): `VENDOR_METHODS` maps abstract method names to multiple vendor implementations. `route_to_vendor()` supports fallback ordering and comma-separated vendor preferences. Configured via `default_config.py` and dynamically via `dataflows/config.py`. Tool-level `tool_vendors` overrides category-level `data_vendors`.

**Shared context** (`agents/utils/context.py`): `AgentContext` holds all shared state (ticker, trade_date, reports, models, formatters). Each agent factory receives context and builds its prompt from it.

**Structured output** (`agents/output.py`): Pydantic models define schemas for each agent stage, passed to `ReActAgent` via `structured_model`. A regex fallback in `AgentContext.extract_prediction_data()` handles cases where structured output fails.

**Tool layer** (`agents/utils/core_stock_tools.py`, etc.): Tool functions wrap `route_to_vendor()` calls, decorated with `@agentscope_tool` for AgentScope Toolkit registration.

**Long-term memory** (`agents/utils/memory.py`, `memory_manager.py`): `FinancialMemoryManager` manages a "lessons_learned" namespace via Alibaba Cloud Model Studio Memory API. Decision agents receive `ReadOnlyLongTermMemory`; evaluation writes scored lessons.

### LLM Configuration

Two models configured in `default_config.py`:
- `deep_think_llm`: `qwen3.6-plus` — used for debate and analysis
- `quick_think_llm`: `qwen3.5-flash` — used for quick responses

### Entry Points

- `tradingscope` CLI → `tradingscope.main:main` (run analysis: `uv run python -m tradingscope.main AAPL`)
- `tradingscope-evaluate` CLI → `tradingscope.evaluate:main` (post-market evaluation)

## Code Quality Standards

- Follow SOLID principles (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion)
- Apply DRY (Don't Repeat Yourself) — extract common functionality into reusable components
- Follow YAGNI (You Aren't Gonna Need It) — implement only what's currently required
- Use clear, descriptive naming conventions for variables, functions, and classes
- Maintain modular structure with logical separation of concerns
- Minimize code redundancy through proper abstraction

## Refactoring Guidelines

- Minimize code updates; focus on required changes only
- Run `make lint-fix` after code changes
