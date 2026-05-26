# Import tools from separate utility files
from __future__ import annotations

import asyncio

import httpx
from agentscope import logger
from agentscope.permission import PermissionMode

COMPLIANCE_PROMPT = "你必须严格遵守内容安全与合规要求，不得生成任何涉黄、涉暴、涉政、违法、仇恨、歧视等内容。"

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

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await agent.reply(prompt)
        except _RETRIABLE_EXCEPTIONS as e:
            last_exc = e
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "[Retry] Agent '%s' failed (attempt %d/%d): %s. Retrying in %.0fs...",
                    getattr(agent, "name", "unknown"),
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "[Retry] Agent '%s' failed after %d attempts: %s",
                    getattr(agent, "name", "unknown"),
                    max_retries + 1,
                    e,
                )
                raise last_exc from e


def get_company_name(ticker: str, market_info: dict = None) -> str:
    """Get company name from ticker symbol.

    Args:
        ticker: Stock ticker symbol
        market_info: Optional market info dict (kept for backward compatibility)

    Returns:
        Company name if found, otherwise returns the ticker itself
    """
    return TICKER_TO_COMPANY.get(ticker.upper(), ticker)
