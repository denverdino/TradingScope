import os
import re

import requests

perplexity_mode = "sonar"  # "sonar-pro"


def _format_response_with_citations(content: str, citations: list[str] | None) -> str:
    """Format response content with inline citation links.

    Args:
        content: The response content with citation markers like [1], [2], etc.
        citations: List of citation URLs

    Returns:
        Formatted content with inline links
    """
    if not citations:
        return content

    # Replace citation markers [n] with inline links [n](url)
    def replace_citation(match):
        idx = int(match.group(1)) - 1  # Citations are 1-indexed
        if 0 <= idx < len(citations):
            return f"[[{idx + 1}]]({citations[idx]})"
        return match.group(0)

    formatted_content = re.sub(r'\[(\d+)\]', replace_citation, content)

    # Append citations list at the end for reference
    citations_section = "\n\n**信息来源:**\n"
    for i, url in enumerate(citations, 1):
        citations_section += f"- [{i}] {url}\n"

    return formatted_content + citations_section


def _call_perplexity_api(messages: list[dict]) -> str:
    """Call Perplexity API with citations enabled.

    Args:
        messages: List of message dicts for the API

    Returns:
        Formatted response with citations
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": perplexity_mode,
        "messages": messages,
    }

    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"]
    citations = data.get("citations", [])

    return _format_response_with_citations(content, citations)


def get_stock_news_perplexity(query, start_date, end_date):
    messages = [
        {
            "role": "user",
            "content": f"Can you search Social Media for {query} from {start_date} to {end_date}? Make sure you only get the data posted during that period.",
        }
    ]
    return _call_perplexity_api(messages)


def get_global_news_perplexity(curr_date, look_back_days=7, limit=5):
    messages = [
        {
            "role": "user",
            "content": f"Can you search global or macroeconomics news from {look_back_days} days before {curr_date} to {curr_date} that would be informative for trading purposes? Make sure you only get the data posted during that period. Limit the results to {limit} articles.",
        }
    ]
    return _call_perplexity_api(messages)


def get_fundamentals_perplexity(ticker, curr_date):
    messages = [
        {
            "role": "user",
            "content": f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc",
        }
    ]
    return _call_perplexity_api(messages)
