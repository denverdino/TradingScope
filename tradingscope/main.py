import argparse
import asyncio
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import markdown

# 导入日志模块
from agentscope import logger

from tradingscope.agents.utils.context import AgentContext
from tradingscope.agents.workflow import analyze
from tradingscope.default_config import DEFAULT_CONFIG


def _configure_memory_debug() -> None:
    """Enable DEBUG logging for memory API when MEMORY_DEBUG env var is set."""
    if not os.getenv("MEMORY_DEBUG"):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"),
    )

    for name in (
        "agentscope_runtime.tools.modelstudio_memory",
        "tradingscope.agents.utils.memory",
    ):
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        log.addHandler(handler)


def send_html_email(subject, html_content, recipient_emails, sender_email, sender_password):
    """Send HTML email via configurable SMTP."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ",".join(recipient_emails)
    part = MIMEText(html_content, "html")
    msg.attach(part)

    # Get SMTP configuration from environment variables with Gmail defaults
    smtp_host = os.getenv("SMTP_SSL_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_SSL_PORT", "465"))

    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_emails, msg.as_string())
        logger.info("Report email sent successfully!")
    except Exception as e:
        logger.error("Failed to send email: %s", e)


def _configure_tool_debug() -> None:
    """Gate tool-related console output behind the TOOL_DEBUG env var.

    When TOOL_DEBUG is *not* set:
      - Suppress the ``system: { "type": "tool_result", ... }`` JSON that
        AgentScope prints for every tool call by replacing
        ``AgentBase._print_last_block`` with a no-op.
    When TOOL_DEBUG *is* set:
      - Keep the original behaviour (the JSON is printed).
    """
    if os.getenv("TOOL_DEBUG"):
        logger.info("TOOL_DEBUG enabled – tool calls will be logged with arguments and results")
        return  # keep default printing behaviour

    # Suppress AgentScope's tool-result JSON output
    from agentscope.agent._agent_base import AgentBase  # noqa: E402

    AgentBase._print_last_block = lambda self, block, msg: None


def main():
    """Main entry point for the TradingScope application."""
    _configure_memory_debug()
    _configure_tool_debug()

    # Create argument parser
    parser = argparse.ArgumentParser(description="TradingScope - Multi-Agents trading framework")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock ticker symbol (e.g., AAPL, BABA)")
    parser.add_argument("--email_to", help="Email address to send the report to")
    parser.add_argument(
        "--output", choices=["markdown", "json", "both"], default="both", help="Output format: markdown (HTML only), json (JSON only), both (default)"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    # Parse arguments
    args = parser.parse_args()

    # Get ticker from command line argument or use default
    ticker = args.ticker

    # Use AgentContext as the single source of truth for trade_date
    trade_date = AgentContext().trade_date
    output = asyncio.run(analyze(ticker))
    final_report = output.report_md
    structured_result = output.structured

    logger.info("******************************* Final Report *******************************")
    logger.info(final_report)

    # Save outputs based on --output flag
    output_mode = args.output

    # Determine local data directory under results_dir
    results_dir = DEFAULT_CONFIG["results_dir"]
    ticker_data_dir = os.path.join(results_dir, "data", trade_date, ticker)
    os.makedirs(ticker_data_dir, exist_ok=True)

    if output_mode in ("markdown", "both"):
        # Generate HTML output from Markdown with table extension
        html_output = markdown.markdown(final_report, extensions=["tables"])

        # Add CSS styling for tables with grid lines
        html_with_style = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Stock Analysis Report: {ticker} ({trade_date})</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
    </style>
</head>
<body>
    {html_output}
</body>
</html>
"""

        # Save HTML output to local data directory
        html_filename = os.path.join(ticker_data_dir, f"{ticker}_report_{trade_date}.html")
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(html_with_style)
        logger.info("HTML report saved to: %s", html_filename)

        # Send email if email address is provided
        if args.email_to:
            sender_email = os.getenv("EMAIL_FROM")
            sender_password = os.getenv("EMAIL_PASSWORD")

            if not sender_email or not sender_password:
                logger.error("EMAIL_FROM and EMAIL_PASSWORD environment variables must be set to send email.")
            else:
                subject = f"Stock Analysis Report: {ticker} ({trade_date})"
                recipient_list = [email.strip() for email in args.email_to.split(",")]
                send_html_email(subject, html_with_style, recipient_list, sender_email, sender_password)

    if output_mode in ("json", "both"):
        # Save each individual structured output as a separate JSON file
        if output.individual_structured:
            for agent_name, json_content in output.individual_structured.items():
                agent_json_filename = os.path.join(ticker_data_dir, f"{agent_name}.json")
                with open(agent_json_filename, "w", encoding="utf-8") as f:
                    f.write(json_content)
                logger.info("Structured output saved to: %s", agent_json_filename)

        # Save combined structured result
        json_filename = os.path.join(ticker_data_dir, f"{ticker}_report_{trade_date}.json")
        json_content = structured_result.to_json()
        with open(json_filename, "w", encoding="utf-8") as f:
            f.write(json_content)
        logger.info("Combined JSON structured report saved to: %s", json_filename)

    if output_mode == "json":
        # When only JSON output is requested, also print it to stdout
        print(json_content)


if __name__ == "__main__":
    main()
