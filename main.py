import asyncio
import os
import sys
from datetime import datetime

from agentscope.model import OpenAIChatModel

from tradingscope.agents.workflow import analyze

# 导入日志模块
from tradingscope.utils.logging_manager import get_logger

logger = get_logger('default')

model = OpenAIChatModel(
    model_name="qwen-max",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    stream=True,
    generate_kwargs={"temperature": 0.1}
)

# Get ticker from command line argument or use default
ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

trade_date = datetime.now().strftime("%Y-%m-%d")

asyncio.run(analyze(model, ticker, trade_date))
