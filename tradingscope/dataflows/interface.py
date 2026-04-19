import logging
import os

# Import from vendor-specific modules
from .alpha_vantage import get_balance_sheet as get_alpha_vantage_balance_sheet
from .alpha_vantage import get_cashflow as get_alpha_vantage_cashflow
from .alpha_vantage import get_fundamentals as get_alpha_vantage_fundamentals
from .alpha_vantage import get_income_statement as get_alpha_vantage_income_statement
from .alpha_vantage import get_indicator as get_alpha_vantage_indicator
from .alpha_vantage import get_insider_transactions as get_alpha_vantage_insider_transactions
from .alpha_vantage import get_news as get_alpha_vantage_news
from .alpha_vantage import get_stock as get_alpha_vantage_stock
from .alpha_vantage_common import AlphaVantageRateLimitError

# Configuration and routing logic
from .config import get_config
from .google import get_google_news
from .local import (
    get_finnhub_company_insider_sentiment,
    get_finnhub_company_insider_transactions,
    get_finnhub_news,
    get_reddit_company_news,
    get_reddit_global_news,
    get_simfin_balance_sheet,
    get_simfin_cashflow,
    get_simfin_income_statements,
    get_YFin_data,
)
from .openai import get_fundamentals_openai, get_global_news_openai, get_stock_news_openai
from .perplexity import get_fundamentals_perplexity, get_global_news_perplexity, get_stock_news_perplexity
from .y_finance import get_balance_sheet as get_yfinance_balance_sheet
from .y_finance import get_cashflow as get_yfinance_cashflow
from .y_finance import get_fundamentals as get_yfinance_fundamentals
from .y_finance import get_income_statement as get_yfinance_income_statement
from .y_finance import get_insider_transactions as get_yfinance_insider_transactions
from .y_finance import get_market_indices as get_yfinance_market_indices
from .y_finance import get_options_analysis as get_yfinance_options_analysis
from .y_finance import get_sector_performance as get_yfinance_sector_performance
from .y_finance import get_stock_stats_indicators_window, get_YFin_data_online, get_YFin_stock_info
from .y_finance import get_volume_analysis as get_yfinance_volume_analysis

logger = logging.getLogger(__name__)

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {"description": "OHLCV stock price data", "tools": ["get_stock_data", "get_stock_info"]},
    "technical_indicators": {"description": "Technical analysis indicators", "tools": ["get_indicators"]},
    "market_context": {
        "description": "Market and sector context data",
        "tools": ["get_sector_performance", "get_market_indices", "get_options_analysis", "get_volume_analysis"],
    },
    "fundamental_data": {"description": "Company fundamentals", "tools": ["get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement"]},
    "news_data": {
        "description": "News (public/insiders, original/processed)",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_sentiment",
            "get_insider_transactions",
        ],
    },
}

VENDOR_LIST = ["local", "yfinance", "openai", "google"]

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
    },
    "get_stock_info": {
        "yfinance": get_YFin_stock_info,
    },
    # market_context
    "get_sector_performance": {
        "yfinance": get_yfinance_sector_performance,
    },
    "get_market_indices": {
        "yfinance": get_yfinance_market_indices,
    },
    "get_options_analysis": {
        "yfinance": get_yfinance_options_analysis,
    },
    "get_volume_analysis": {
        "yfinance": get_yfinance_volume_analysis,
    },
    # technical_indicators
    "get_indicators": {
        "yfinance": get_stock_stats_indicators_window,
        "alpha_vantage": get_alpha_vantage_indicator,
    },
    # fundamental_data
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
        # "openai": get_fundamentals_openai,
        "perplexity": get_fundamentals_perplexity,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
    },
    # news_data
    "get_news": {
        "google": get_google_news,
        "alpha_vantage": get_alpha_vantage_news,
        "perplexity": get_stock_news_perplexity,
        # "openai": get_stock_news_openai,
    },
    "get_global_news": {
        "perplexity": get_global_news_perplexity,
        # "openai": get_global_news_openai,
    },
    "get_insider_sentiment": {"local": get_finnhub_company_insider_sentiment},
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
    },
}


def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")


def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")


def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)

    # Handle comma-separated vendors
    primary_vendors = [v.strip() for v in vendor_config.split(",")]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Get all available vendors for this method for fallback
    all_available_vendors = list(VENDOR_METHODS[method].keys())

    # Create fallback vendor list: primary vendors first, then remaining vendors as fallbacks
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    # Debug: Print fallback ordering
    primary_str = " → ".join(primary_vendors)
    fallback_str = " → ".join(fallback_vendors)
    logger.debug("%s - Primary: [%s] | Full fallback order: [%s]", method, primary_str, fallback_str)

    # Track results and execution state
    results = []
    vendor_attempt_count = 0
    any_primary_vendor_attempted = False
    successful_vendor = None

    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            if vendor in primary_vendors:
                logger.info("Vendor '%s' not supported for method '%s', falling back to next vendor", vendor, method)
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        is_primary_vendor = vendor in primary_vendors
        vendor_attempt_count += 1

        # Track if we attempted any primary vendor
        if is_primary_vendor:
            any_primary_vendor_attempted = True

        # Debug: Print current attempt
        vendor_type = "PRIMARY" if is_primary_vendor else "FALLBACK"
        logger.debug("Attempting %s vendor '%s' for %s (attempt #%d)", vendor_type, vendor, method, vendor_attempt_count)

        # Handle list of methods for a vendor
        if isinstance(vendor_impl, list):
            vendor_methods = [(impl, vendor) for impl in vendor_impl]
            logger.debug("Vendor '%s' has multiple implementations: %d functions", vendor, len(vendor_methods))
        else:
            vendor_methods = [(vendor_impl, vendor)]

        # Run methods for this vendor
        vendor_results = []
        for impl_func, vendor_name in vendor_methods:
            try:
                logger.debug("Calling %s from vendor '%s'...", impl_func.__name__, vendor_name)
                result = impl_func(*args, **kwargs)
                vendor_results.append(result)
                logger.debug("%s from vendor '%s' completed successfully", impl_func.__name__, vendor_name)

            except AlphaVantageRateLimitError as e:
                if vendor == "alpha_vantage":
                    logger.warning("Alpha Vantage rate limit exceeded, falling back to next available vendor")
                    logger.debug("Rate limit details: %s", e)
                # Continue to next vendor for fallback
                continue
            except Exception as e:
                # Log error but continue with other implementations
                logger.warning("%s from vendor '%s' failed: %s", impl_func.__name__, vendor_name, e)
                continue

        # Add this vendor's results
        if vendor_results:
            results.extend(vendor_results)
            successful_vendor = vendor
            logger.debug("Vendor '%s' succeeded - Got %d result(s)", vendor, len(vendor_results))

            # Stopping logic: Stop after first successful vendor for single-vendor configs
            # Multiple vendor configs (comma-separated) may want to collect from multiple sources
            if len(primary_vendors) == 1:
                logger.debug("Stopping after successful vendor '%s' (single-vendor config)", vendor)
                break
        else:
            logger.warning("Vendor '%s' produced no results", vendor)

    # Final result summary
    if not results:
        logger.error("All %d vendor attempts failed for method '%s'", vendor_attempt_count, method)
        raise RuntimeError(f"All vendor implementations failed for method '{method}'")
    else:
        logger.debug("Method '%s' completed with %d result(s) from %d vendor attempt(s)", method, len(results), vendor_attempt_count)

    # Return single result if only one, otherwise concatenate as string
    if len(results) == 1:
        return results[0]
    else:
        # Convert all results to strings and concatenate
        return "\n".join(str(result) for result in results)
