#!/usr/bin/env python3
"""Example script demonstrating how to use the China Market Analyst Agent."""

import asyncio
import datetime
import os

from agentscope.model import OpenAIChatModel

from tradingscope.agents.analysts.china_market_analyst import (
    create_china_market_analyst_agent,
    create_china_stock_screener_agent,
    run_china_market_analyst,
    run_china_stock_screener,
)


async def main():

    model = OpenAIChatModel(
        model_name="qwen-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
    )

    trade_date = datetime.now().strftime("%Y-%m-%d")

    analyst = create_china_market_analyst_agent(model)
    screener = create_china_stock_screener_agent(model)

    # 运行分析师
    res1 = await run_china_market_analyst(analyst, trade_date=trade_date, ticker="600519")
    # 运行筛选器
    res2 = await run_china_stock_screener(screener, trade_date=trade_date)

    print("Analyst Output:\n", res1.content)
    print("Screener Output:\n", res2.content)


if __name__ == "__main__":
    asyncio.run(main())
