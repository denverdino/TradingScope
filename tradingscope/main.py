import argparse
import asyncio
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import markdown

# 导入日志模块
from agentscope import logger
from agentscope.model import OpenAIChatModel

from tradingscope.agents.workflow import analyze


def send_html_email(subject, html_content, recipient_emails, sender_email, sender_password):
    """Send HTML email via configurable SMTP."""
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = ','.join(recipient_emails)
    part = MIMEText(html_content, 'html')
    msg.attach(part)

    # Get SMTP configuration from environment variables with Gmail defaults
    smtp_host = os.getenv('SMTP_SSL_HOST', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_SSL_PORT', '465'))

    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_emails, msg.as_string())
        print("Report email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


def main():
    """Main entry point for the TradingScope application."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='TradingScope - Multi-Agents trading framework')
    parser.add_argument('ticker', nargs='?', default='AAPL', help='Stock ticker symbol (e.g., AAPL, BABA)')
    parser.add_argument('--email_to', help='Email address to send the report to')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    # Parse arguments
    args = parser.parse_args()

    # Initialize model
    # extra_body with enable_thinking=True enables Qwen3 reasoning/thinking mode
    model = OpenAIChatModel(
        model_name="qwen3-max-preview",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=True,
        generate_kwargs={
            "temperature": 0.1,
            "extra_body": {"enable_thinking": True}
        }
    )

    # Get ticker from command line argument or use default
    ticker = args.ticker

    trade_date = datetime.now().strftime("%Y-%m-%d")
    final_report = asyncio.run(analyze(model, ticker, trade_date))

    # Generate HTML output from Markdown with table extension
    html_output = markdown.markdown(final_report, extensions=['tables'])

    # Add CSS styling for tables with grid lines
    html_with_style = f"""
<!DOCTYPE html>
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

    print("******************************* Final Report *******************************")
    print(final_report)

    # Save HTML output to a file
    html_filename = f"{ticker}_report_{trade_date}.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_with_style)
    print(f"\nHTML report saved to: {html_filename}")

    # Send email if email address is provided
    if args.email_to:
        sender_email = os.getenv("EMAIL_FROM")
        sender_password = os.getenv("EMAIL_PASSWORD")

        if not sender_email or not sender_password:
            print("Error: EMAIL_FROM and EMAIL_PASSWORD environment variables must be set to send email.")
        else:
            subject = f"Stock Analysis Report: {ticker} ({trade_date})"
            recipient_list = [email.strip() for email in args.email_to.split(',')]
            send_html_email(subject, html_with_style, recipient_list, sender_email, sender_password)

if __name__ == "__main__":
    main()
