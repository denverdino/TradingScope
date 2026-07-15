#!/usr/bin/env python3
"""Example script demonstrating the full analysis workflow with structured output."""

import asyncio

from tradingscope.agents.renderers import render_full_report
from tradingscope.agents.workflow import analyze


async def main():
    result = await analyze("MSFT")
    print("=" * 60)
    print("Final Markdown Report")
    print(render_full_report(result))
    print("=" * 60)

    print("\nStructured JSON Output:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
