import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": os.path.join(os.path.expanduser("~"), "Documents", "TradingAgents", "data"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "dashscope",
    "deep_think_llm": "qwen3.7-max",
    "quick_think_llm": "qwen3.6-flash",
    "backend_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "builtin_tools_model": "qwen3.7-plus",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "data_vendors": {
        "core_stock_apis": "yfinance",  # Options: yfinance, alpha_vantage, local
        "fundamental_data": "yfinance",  # Options: yfinance, alpha_vantage, local (dashscope: experimental, not yet wired)
        "news_data": "alpha_vantage",  # Options: perplexity, alpha_vantage, google, local (dashscope: experimental)
        "technical_indicators": "yfinance",
        "market_context": "yfinance",  # Options: yfinance (sector performance, market indices)
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
        # Example: "get_news": "dashscope",               # Override category default
    },
}
