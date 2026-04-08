#!/usr/bin/env python3
"""Test script for the workflow"""

import asyncio

from tradingscope.agents.workflow import analyze


async def main():
    report = await analyze("AAPL")
    print("Final Report")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
