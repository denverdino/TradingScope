import argparse
import asyncio
import os
from datetime import datetime

from agentscope.model import OpenAIChatModel

from tradingscope.agents.workflow import analyze

# 导入日志模块
from tradingscope.utils.logging_manager import get_logger

logger = get_logger('default')

def main():
    """Main entry point for the TradingScope application."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='TradingScope - Multi-Agents trading framework')
    parser.add_argument('ticker', nargs='?', default='AAPL', help='Stock ticker symbol (e.g., AAPL, BABA)')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    # Parse arguments
    args = parser.parse_args()

    # Initialize model
    model = OpenAIChatModel(
        model_name="qwen-max",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
        generate_kwargs={"temperature": 0.1}
    )

    # Get ticker from command line argument or use default
    ticker = args.ticker

    trade_date = datetime.now().strftime("%Y-%m-%d")
    asyncio.run(analyze(model, ticker, trade_date))

if __name__ == "__main__":
    main()
