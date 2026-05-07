"""DashScope-native dataflow functions for web-search-based data retrieval.

Replaces the former openai.py module which used the OpenAI Responses API
(web_search_preview tool). These functions use dashscope.Generation.call
with the `plugins="web_search"` flag to invoke DashScope's built-in web
search capability, and `result_format="message"` for structured output.
"""

import logging
import os
from typing import Optional

import dashscope
from dashscope.common.error import DashScopeException

from .config import get_config

logger = logging.getLogger(__name__)


def _extract_dashscope_content(response) -> str:
    """Extract text content from a DashScope Generation response.

    Returns empty string if response structure is malformed.
    """
    if not hasattr(response, "output") or response.output is None:
        logger.warning("[dashscope] Response missing 'output' field")
        return ""
    choices = getattr(response.output, "choices", None)
    if not choices or len(choices) == 0:
        logger.warning("[dashscope] Response has no choices")
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        logger.warning("[dashscope] First choice has no 'message'")
        return ""
    return getattr(message, "content", "") or ""


def get_stock_news_dashscope(query: str, start_date: str, end_date: str) -> str:
    """Search social media for stock-related news via DashScope web search.

    Args:
        query: Stock ticker or company name to search for.
        start_date: Start date for the search period (YYYY-MM-DD).
        end_date: End date for the search period (YYYY-MM-DD).

    Returns:
        DashScope web search result as raw text.

    Raises:
        RuntimeError: If DASHSCOPE_API_KEY is not set, or if DashScope API
            returns a non-200 status code, or if the SDK call fails.
    """
    config = get_config()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY environment variable not set")

    prompt = f"Can you search Social Media for {query} from {start_date} to {end_date}? Make sure you only get the data posted during that period."

    try:
        response = dashscope.Generation.call(
            api_key=api_key,
            model=config["quick_think_llm"],
            messages=[{"role": "user", "content": prompt}],
            plugins="web_search",  # DashScope built-in web search plugin
            result_format="message",
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope API error: status={response.status_code}, code={response.code}, message={response.message}"
            )

        content = _extract_dashscope_content(response)
        if not content:
            logger.warning("[dashscope] Empty response content for stock news query")
            return ""
        return content
    except (DashScopeException, ConnectionError, TimeoutError) as e:
        logger.warning("[dashscope] SDK/network error fetching stock news: %s", e)
        raise RuntimeError(f"DashScope call failed: {e}") from e


def get_global_news_dashscope(curr_date: str, look_back_days: int = 7, limit: int = 5) -> str:
    """Search global/macro economics news via DashScope web search.

    Args:
        curr_date: Current date as reference point (YYYY-MM-DD).
        look_back_days: Number of days to look back from curr_date. Defaults to 7.
        limit: Maximum number of articles to return. Defaults to 5.

    Returns:
        DashScope web search result as raw text.

    Raises:
        RuntimeError: If DASHSCOPE_API_KEY is not set, or if DashScope API
            returns a non-200 status code, or if the SDK call fails.
    """
    config = get_config()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY environment variable not set")

    prompt = (
        f"Can you search global or macroeconomics news from {look_back_days} days before {curr_date} to {curr_date}"
        f" that would be informative for trading purposes?"
        f" Make sure you only get the data posted during that period."
        f" Limit the results to {limit} articles."
    )

    try:
        response = dashscope.Generation.call(
            api_key=api_key,
            model=config["quick_think_llm"],
            messages=[{"role": "user", "content": prompt}],
            plugins="web_search",  # DashScope built-in web search plugin
            result_format="message",
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope API error: status={response.status_code}, code={response.code}, message={response.message}"
            )

        content = _extract_dashscope_content(response)
        if not content:
            logger.warning("[dashscope] Empty response content for global news query")
            return ""
        return content
    except (DashScopeException, ConnectionError, TimeoutError) as e:
        logger.warning("[dashscope] SDK/network error fetching global news: %s", e)
        raise RuntimeError(f"DashScope call failed: {e}") from e


def get_fundamentals_dashscope(ticker: str, curr_date: str) -> str:
    """Search company fundamentals via DashScope web search.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        curr_date: Reference date (YYYY-MM-DD) for the search period.

    Returns:
        DashScope web search result as raw text.

    Raises:
        RuntimeError: If DASHSCOPE_API_KEY is not set, or if DashScope API
            returns a non-200 status code, or if the SDK call fails.
    """
    config = get_config()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY environment variable not set")

    prompt = (
        f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date}"
        f" to the month of {curr_date}."
        f" Make sure you only get the data posted during that period."
        f" List as a table, with PE/PS/Cash flow/ etc"
    )

    try:
        response = dashscope.Generation.call(
            api_key=api_key,
            model=config["quick_think_llm"],
            messages=[{"role": "user", "content": prompt}],
            plugins="web_search",  # DashScope built-in web search plugin
            result_format="message",
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope API error: status={response.status_code}, code={response.code}, message={response.message}"
            )

        content = _extract_dashscope_content(response)
        if not content:
            logger.warning("[dashscope] Empty response content for fundamentals query")
            return ""
        return content
    except (DashScopeException, ConnectionError, TimeoutError) as e:
        logger.warning("[dashscope] SDK/network error fetching fundamentals: %s", e)
        raise RuntimeError(f"DashScope call failed: {e}") from e
