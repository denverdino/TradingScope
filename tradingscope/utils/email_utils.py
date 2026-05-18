import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from agentscope import logger


def send_html_email(subject, html_content, recipient_emails, sender_email, sender_password):
    """Send HTML email via configurable SMTP."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ",".join(recipient_emails)
    part = MIMEText(html_content, "html")
    msg.attach(part)

    smtp_host = os.getenv("SMTP_SSL_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_SSL_PORT", "465"))

    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_emails, msg.as_string())
        logger.info("Report email sent successfully!")
    except Exception as e:
        logger.error("Failed to send email: %s", e)
