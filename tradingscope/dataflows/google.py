from typing import Annotated

from tradingscope.agents.utils.agent_utils import get_company_name

from .googlenews_utils import getNewsData


def get_google_news(
    ticker: Annotated[str, "Stock ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch Google News for a stock using company name.

    Args:
        ticker: Stock ticker symbol (will be converted to company name)
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted news string with inline links
    """
    # Convert ticker to company name for better search results
    company_name = get_company_name(ticker)
    query = company_name.replace(" ", "+")

    news_results = getNewsData(query, start_date, end_date)

    if len(news_results) == 0:
        return ""

    news_str = ""

    for news in news_results:
        title = news.get('title', '无标题')
        link = news.get('link', '')
        source = news.get('source', '未知来源')
        snippet = news.get('snippet', '')
        date = news.get('date', '')

        # Build article entry with inline link
        if link:
            news_str += f"### [{title}]({link})\n"
        else:
            news_str += f"### {title}\n"

        news_str += f"**来源:** {source}"
        if date:
            news_str += f" | **时间:** {date}"
        news_str += "\n\n"

        if snippet:
            news_str += f"{snippet}\n\n"

        news_str += "---\n\n"

    return f"## {company_name} ({ticker}) Google News ({start_date} 至 {end_date}):\n\n{news_str}"
