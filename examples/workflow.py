#!/usr/bin/env python3
"""Example script demonstrating the full analysis workflow with structured output."""

import asyncio

from tradingscope.agents.workflow import analyze


async def main():
    output = await analyze("MSFT")
    print("=" * 60)
    print("Final Markdown Report")
    print(output.report_md)
    print("=" * 60)

    print("\nStructured JSON Output:")
    print(output.structured.to_json())


if __name__ == "__main__":
    asyncio.run(main())
