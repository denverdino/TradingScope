import os

from openai import OpenAI

perplexity_mode = "sonar" #"sonar-pro"

def get_stock_news_perplexity(query, start_date, end_date):
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )

    response = client.chat.completions.create(
        model=perplexity_mode,
        messages=[
            {
                "role": "user",
                "content": f"Can you search Social Media for {query} from {start_date} to {end_date}? Make sure you only get the data posted during that period.",
            }
        ]
    )

    return response.choices[0].message.content


def get_global_news_perplexity(curr_date, look_back_days=7, limit=5):
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )

    response = client.chat.completions.create(
        model=perplexity_mode,
        messages=[
            {
                "role": "user",
                "content": f"Can you search global or macroeconomics news from {look_back_days} days before {curr_date} to {curr_date} that would be informative for trading purposes? Make sure you only get the data posted during that period. Limit the results to {limit} articles.",
            }
        ]
    )
    return response.choices[0].message.content


def get_fundamentals_perplexity(ticker, curr_date):
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )

    response = client.chat.completions.create(
        model=perplexity_mode,
        messages=[
            {
                "role": "user",
                "content": f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc",
            }
        ]
    )

    return response.choices[0].message.content
