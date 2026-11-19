# Import tools from separate utility files
from __future__ import annotations

import asyncio

import httpx
from agentscope import logger
from agentscope.event import (
    ModelCallEndEvent,
    ReplyStartEvent,
    TextBlockDeltaEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
)
from agentscope.message import Msg
from agentscope.permission import PermissionMode

from .local_audit import get_audit_recorder, redact

COMPLIANCE_PROMPT = "你必须严格遵守内容安全与合规要求，不得生成任何涉黄、涉暴、涉政、违法、仇恨、歧视等内容。"

ANALYST_DRAFT_OUTPUT_REQUIREMENTS = """分析素材必须覆盖：
- 唯一方向判断（看涨/看跌/中性）、唯一操作建议（买入/持有/卖出）、置信度（0-1）及核心理由，方向与操作不得矛盾。
- 每条关键证据的来源、原始链接（如有）、数据日期，区分事实与推断；保留最强反证。
- 有依据的价格及计价货币、适用条件和失效条件；价格无依据时留空（null），不得为补齐结构而编造。
- 数据缺失与适用限制，以及它们对结论可信度的影响。
只输出中文分析素材，不要输出 JSON、JSON 代码块或固定 Markdown 报告模板；下一阶段根据严格 schema 生成正式结果。"""

DEBATE_OUTPUT_RULES = (
    "发言要求：最多2个关键论点，每个论点注明来源与数据时点；不得重抄共享报告，只引用支撑论点所需的关键数据；"
    "只回应已收到的发言，不预设他人尚未提出的观点；他人证据成立时可以承认并修正己方判断。"
)
DEBATE_FOLLOW_UP_RULES = "本轮仅补充新增证据、观点修订和核心分歧，不重复已陈述的内容。"
DEBATE_FINAL_ROUND_RULES = (
    "这是最后一轮发言：请收敛为己方仍然成立的最强论据、他人提出的最强反证、剩余核心分歧，以及撤回或调整立场的条件；不要求己方获胜，结论应服从证据。"
)


def build_debate_round_prompt(round_num: int, max_rounds: int, *, opening: str, rebuttal: str) -> str:
    parts = [opening if round_num == 1 else rebuttal, DEBATE_OUTPUT_RULES]
    if round_num > 1:
        parts.append(DEBATE_FOLLOW_UP_RULES)
    if round_num == max_rounds:
        parts.append(DEBATE_FINAL_ROUND_RULES)
    return "\n".join(parts)


# Mapping of ticker symbols to company names
TICKER_TO_COMPANY = {
    # Tech Giants
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "GOOG": "Google",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "Nvidia",
    "TSM": "Taiwan Semiconductor Manufacturing Company",
    "AVGO": "Broadcom",
    "ASML": "ASML",
    "AMD": "AMD",
    "INTC": "Intel",
    "QCOM": "Qualcomm",
    "MU": "Micron",
    # Cloud & Software
    "CRM": "Salesforce",
    "ORCL": "Oracle",
    "ADBE": "Adobe",
    "NOW": "ServiceNow",
    "SNOW": "Snowflake",
    "DDOG": "Datadog",
    "MDB": "MongoDB",
    "NET": "Cloudflare",
    "TEAM": "Atlassian",
    "SHOP": "Shopify",
    "TWLO": "Twilio",
    # Cybersecurity
    "CRWD": "CrowdStrike",
    "PANW": "Palo Alto Networks",
    "ZS": "Zscaler",
    # Consumer Tech
    "TSLA": "Tesla",
    "NFLX": "Netflix",
    "SPOT": "Spotify",
    "ROKU": "Roku",
    "ZM": "Zoom",
    "SNAP": "Snap Inc",
    "PINS": "Pinterest",
    "UBER": "Uber",
    "LYFT": "Lyft",
    # Fintech & Payments
    "V": "Visa",
    "MA": "Mastercard",
    "PYPL": "PayPal",
    "SQ": "Block Square",
    "COIN": "Coinbase",
    "HOOD": "Robinhood",
    # Finance
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "WFC": "Wells Fargo",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "C": "Citigroup",
    # Healthcare & Pharma
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth",
    "PFE": "Pfizer",
    "MRNA": "Moderna",
    "ABBV": "AbbVie",
    "LLY": "Eli Lilly",
    "BMY": "Bristol-Myers Squibb",
    "GILD": "Gilead Sciences",
    "AMGN": "Amgen",
    "BIIB": "Biogen",
    "REGN": "Regeneron",
    "VRTX": "Vertex Pharmaceuticals",
    "CVS": "CVS Health",
    # Retail & Consumer
    "WMT": "Walmart",
    "COST": "Costco",
    "TGT": "Target",
    "HD": "Home Depot",
    "LOW": "Lowe's",
    "NKE": "Nike",
    "SBUX": "Starbucks",
    "MCD": "McDonald's",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "DIS": "Disney",
    # Chinese Tech
    "BABA": "Alibaba",
    "JD": "JD.com",
    "PDD": "Pinduoduo PDD",
    "BIDU": "Baidu",
    "NIO": "NIO",
    "XPEV": "XPeng",
    "LI": "Li Auto",
    # Automotive
    "F": "Ford",
    "GM": "General Motors",
    "TM": "Toyota",
    "HMC": "Honda",
    "RIVN": "Rivian",
    "LCID": "Lucid Motors",
    # Aerospace & Defense
    "BA": "Boeing",
    "LMT": "Lockheed Martin",
    "RTX": "Raytheon",
    "NOC": "Northrop Grumman",
    # Telecom
    "T": "AT&T",
    "VZ": "Verizon",
    "TMUS": "T-Mobile",
    # Networking
    "CSCO": "Cisco",
    # Other
    "PLTR": "Palantir",
    "X": "Twitter X",
    "SQSP": "Squarespace",
}

# Backward compatibility alias
ticker_to_company = TICKER_TO_COMPANY


_RETRIABLE_EXCEPTIONS = (
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.ConnectError,
)


async def call_agent_with_retry(
    agent,
    prompt,
    max_retries: int = 3,
    base_delay: float = 2.0,
):
    """Call an agent with retry on transient network errors.

    Uses exponential backoff to retry on httpx connection/protocol errors
    that commonly occur during streaming LLM responses.

    Args:
        agent: AgentScope Agent to call
        prompt: Message prompt to pass to the agent (or None)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff

    Returns:
        Agent response Msg

    Raises:
        The last exception if all retries are exhausted
    """
    # Auto-allow all tool calls (no user confirmation needed in batch mode)
    if hasattr(agent, "state") and hasattr(agent.state, "permission_context"):
        agent.state.permission_context.mode = PermissionMode.BYPASS

    agent_name = getattr(agent, "name", "unknown")
    audit = get_audit_recorder()

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            final_msg: Msg | None = None
            text_buf: list[str] = []
            tool_result_buf: dict[str, list[str]] = {}
            async for event in agent._reply(prompt):
                if audit is not None:
                    try:
                        await audit.observe(agent, event, attempt)
                    except Exception:
                        logger.warning("Local audit event could not be recorded")
                if isinstance(event, Msg):
                    final_msg = event
                elif isinstance(event, ReplyStartEvent):
                    logger.info("[%s] started", agent_name)
                elif isinstance(event, TextBlockDeltaEvent):
                    text_buf.append(event.delta)
                elif isinstance(event, ToolCallStartEvent):
                    if text_buf:
                        logger.info("[%s] %s", agent_name, redact("".join(text_buf)))
                        text_buf.clear()
                    logger.info("[%s] -> tool: %s", agent_name, event.tool_call_name)
                elif isinstance(event, ToolResultStartEvent):
                    tool_result_buf[event.tool_call_id] = []
                elif isinstance(event, ToolResultTextDeltaEvent):
                    tool_result_buf.setdefault(event.tool_call_id, []).append(event.delta)
                elif isinstance(event, ToolResultEndEvent):
                    result_text = "".join(tool_result_buf.pop(event.tool_call_id, []))
                    logger.debug("[%s] tool result [%s]: %s", agent_name, event.tool_call_id, redact(result_text)[:500])
                elif isinstance(event, ModelCallEndEvent):
                    if text_buf:
                        logger.info("[%s] %s", agent_name, redact("".join(text_buf)))
                        text_buf.clear()
                    logger.info("[%s] tokens: in=%d out=%d", agent_name, event.input_tokens, event.output_tokens)
            if final_msg is None:
                raise RuntimeError(f"Agent '{agent_name}' did not produce a final message.")
            if audit is not None:
                audit.record("agent_response", agent=agent_name, phase="draft", attempt=attempt, text=final_msg.get_text_content())
            return final_msg
        except _RETRIABLE_EXCEPTIONS as e:
            last_exc = e
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "[Retry] Agent '%s' failed (attempt %d/%d): %s. Retrying in %.0fs...",
                    agent_name,
                    attempt + 1,
                    max_retries,
                    redact(str(e)),
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "[Retry] Agent '%s' failed after %d attempts: %s",
                    agent_name,
                    max_retries + 1,
                    redact(str(e)),
                )
                raise last_exc from e
        finally:
            if audit is not None:
                audit.finish(agent, attempt)


def get_company_name(ticker: str, market_info: dict = None) -> str:
    """Get company name from ticker symbol.

    Args:
        ticker: Stock ticker symbol
        market_info: Optional market info dict (kept for backward compatibility)

    Returns:
        Company name if found, otherwise returns the ticker itself
    """
    return TICKER_TO_COMPANY.get(ticker.upper(), ticker)
