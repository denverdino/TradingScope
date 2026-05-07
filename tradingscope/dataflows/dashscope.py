import os

import dashscope

from .config import get_config


def get_stock_news_dashscope(query, start_date, end_date):
    config = get_config()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    prompt = f"Can you search Social Media for {query} from {start_date} to {end_date}? Make sure you only get the data posted during that period."

    response = dashscope.Generation.call(
        api_key=api_key,
        model=config["quick_think_llm"],
        messages=[{"role": "user", "content": prompt}],
        plugins="web_search",
        result_format="message",
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    raise RuntimeError(f"DashScope API error: status={response.status_code}, message={response.message}")


def get_global_news_dashscope(curr_date, look_back_days=7, limit=5):
    config = get_config()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    prompt = (
        f"Can you search global or macroeconomics news from {look_back_days} days before {curr_date} to {curr_date}"
        f" that would be informative for trading purposes?"
        f" Make sure you only get the data posted during that period."
        f" Limit the results to {limit} articles."
    )

    response = dashscope.Generation.call(
        api_key=api_key,
        model=config["quick_think_llm"],
        messages=[{"role": "user", "content": prompt}],
        plugins="web_search",
        result_format="message",
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    raise RuntimeError(f"DashScope API error: status={response.status_code}, message={response.message}")


def get_fundamentals_dashscope(ticker, curr_date):
    config = get_config()
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    prompt = (
        f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date}"
        f" to the month of {curr_date}."
        f" Make sure you only get the data posted during that period."
        f" List as a table, with PE/PS/Cash flow/ etc"
    )

    response = dashscope.Generation.call(
        api_key=api_key,
        model=config["quick_think_llm"],
        messages=[{"role": "user", "content": prompt}],
        plugins="web_search",
        result_format="message",
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    raise RuntimeError(f"DashScope API error: status={response.status_code}, message={response.message}")
