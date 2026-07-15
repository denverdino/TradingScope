import argparse
import asyncio
import os

import markdown
from agentscope import logger

from tradingscope.agents.renderers import render_full_report
from tradingscope.agents.workflow import analyze
from tradingscope.default_config import DEFAULT_CONFIG
from tradingscope.utils.email_utils import send_html_email


def main():
    """Main entry point for the TradingScope application."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="TradingScope - Multi-Agents trading framework")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Stock ticker symbol (e.g., AAPL, BABA)")
    parser.add_argument("--email-to", help="Email address to send the report to")
    parser.add_argument(
        "--output", choices=["markdown", "json", "both"], default="both", help="Output format: markdown (HTML only), json (JSON only), both (default)"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    # Parse arguments
    args = parser.parse_args()

    # Get ticker from command line argument or use default
    ticker = args.ticker

    result = asyncio.run(analyze(ticker))
    trade_date = result.trade_date.isoformat()
    final_report = render_full_report(result)

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
        node_outputs = {
            "market_analyst": result.analysts.market,
            "fundamentals_analyst": result.analysts.fundamentals,
            "news_analyst": result.analysts.news,
            "social_media_analyst": result.analysts.social_media,
            "research_manager": result.research_manager,
            "trader": result.trader,
            "portfolio_manager": result.portfolio_manager,
        }
        for agent_name, output in node_outputs.items():
            agent_json_filename = os.path.join(ticker_data_dir, f"{agent_name}.json")
            with open(agent_json_filename, "w", encoding="utf-8") as f:
                f.write(output.model_dump_json(indent=2))
            logger.info("Structured output saved to: %s", agent_json_filename)

        # Save combined structured result
        json_filename = os.path.join(ticker_data_dir, f"{ticker}_report_{trade_date}.json")
        json_content = result.model_dump_json(indent=2)
        with open(json_filename, "w", encoding="utf-8") as f:
            f.write(json_content)
        logger.info("Combined JSON structured report saved to: %s", json_filename)

    if output_mode == "json":
        # When only JSON output is requested, also print it to stdout
        print(json_content)


if __name__ == "__main__":
    main()
