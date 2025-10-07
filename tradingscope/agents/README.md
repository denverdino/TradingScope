# Trading Agents Documentation

This project includes several trading agents built with the AgentScope framework.

## Risk Management Debate System

The risk management debate system implements a multi-agent debate pattern where three risk analysts with different perspectives debate investment decisions:

### Agents

1. **Aggressive Risk Debator**: Advocates for high-return, high-risk investment opportunities
2. **Conservative Risk Debator**: Focuses on protecting assets and minimizing volatility
3. **Neutral Risk Debator**: Provides a balanced perspective weighing potential gains and risks
4. **Risk Manager**: Evaluates the debate and makes the final decision

### Architecture

The system follows the AgentScope multi-agent debate pattern:

```
┌─────────────────────┐
│  Debate Orchestrator │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   MsgHub (Shared)   │
└───┬───────┬───────┬─┘
    │       │       │
    ▼       ▼       ▼
┌───────┐ ┌───────┐ ┌───────┐
│Aggr.  │ │Cons.  │ │Neutral│
│Debator│ │Debator│ │Debator│
└───────┘ └───────┘ └───────┘
    │       │       │
    └───┬───┴───┬───┘
        │       │
        ▼       ▼
┌─────────────┐ ┌─────────────┐
│ Risk        │ │ Risk        │
│ Manager     │ │ Manager     │
│ (Final      │ │ (Memory)    │
│ Decision)   │ │             │
└─────────────┘ └─────────────┘
```

### Usage

```python
from tradingscope.agents.risk_mgmt.debate_orchestrator import create_debate_orchestrator

# Model configuration
model_config = {
    "model_name": "gpt-4o-mini",
    "api_key": "your-api-key",
}

# Create orchestrator
orchestrator = create_debate_orchestrator(model_config, max_rounds=3)

# Run debate
final_decision = await orchestrator.run_debate(
    company_name="TSLA",
    trader_plan="Plan to buy 200 shares of TSLA",
    market_research_report="Market analysis data...",
    sentiment_report="Social media sentiment...",
    news_report="Latest world affairs...",
    fundamentals_report="Company fundamentals..."
)
```

## Fundamentals Analyst Agent

Analyzes stock fundamentals and provides investment insights.

See [fundamentals_analyst.py](../agents/analysts/fundamentals_analyst.py) for implementation details.