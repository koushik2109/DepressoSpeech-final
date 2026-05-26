"""Email service for sending OTP verification emails.

Fixed: Uses asyncio.to_thread() for non-blocking SMTP operations.
"""

import asyncio
import logging
import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config.settings import get_settings

logger = logging.getLogger("mindscope.email")
settings = get_settings()


def generate_otp(length: int = 6) -> str:
    """Generate a random numeric OTP."""
    return "".join(random.choices(string.digits, k=length))


def _send_smtp(to_email: str, msg: MIMEMultipart) -> bool:
    """Synchronous SMTP send — called via asyncio.to_thread().

    Uses SMTP_SSL (port 465) by default which works on Render's infrastructure.
    Falls back to STARTTLS (port 587) when SMTP_PORT is explicitly set to 587.
    """
    port = int(settings.SMTP_PORT or 465)
    try:
        if port == 465:
            with smtplib.SMTP_SSL(settings.SMTP_HOST, port, timeout=8) as server:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(settings.SMTP_USER, to_email, msg.as_string())
        else:
            with smtplib.SMTP(settings.SMTP_HOST, port, timeout=8) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(settings.SMTP_USER, to_email, msg.as_string())
        logger.info(f"[EMAIL] Email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"[EMAIL] Failed to send email to {to_email}: {e}")
        return False


def _html_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_card_html(title: str, intro: str, rows: list[tuple[str, object]] | None = None, footer: str = "") -> str:
    rows_html = ""
    for label, value in rows or []:
        display = _html_escape(value if value not in (None, "") else "Not provided")
        rows_html += f"""
            <tr>
                <td class="label">{_html_escape(label)}</td>
                <td class="value">{display}</td>
            </tr>
        """

    table_html = f'<table class="details" role="presentation">{rows_html}</table>' if rows_html else ""
    footer_text = _html_escape(footer or "This is an automated MindScope message.")
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ margin: 0; padding: 0; background: #f4faf7; font-family: Arial, Helvetica, sans-serif; color: #1b1b1b; }}
            .wrap {{ padding: 36px 16px; }}
            .card {{ max-width: 560px; margin: 0 auto; background: #ffffff; border: 1px solid #dcebe3; border-radius: 18px; overflow: hidden; box-shadow: 0 12px 36px rgba(27, 58, 45, 0.10); }}
            .header {{ background: #1b3a2d; padding: 28px 32px; text-align: center; }}
            .brand {{ margin: 0; color: #ffffff; font-size: 22px; font-weight: 800; letter-spacing: 0.2px; }}
            .subtitle {{ margin: 6px 0 0; color: #b7e4c7; font-size: 13px; font-weight: 600; }}
            .body {{ padding: 30px 32px 28px; }}
            h1 {{ margin: 0 0 12px; color: #1b1b1b; font-size: 22px; line-height: 1.3; font-weight: 800; }}
            .intro {{ margin: 0 0 24px; color: #4f5f57; font-size: 15px; line-height: 1.6; }}
            .details {{ width: 100%; border-collapse: collapse; border: 1px solid #e4eee8; border-radius: 12px; overflow: hidden; }}
            .details tr + tr td {{ border-top: 1px solid #e4eee8; }}
            .label {{ width: 38%; padding: 13px 14px; background: #f4faf7; color: #2d6a4f; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; }}
            .value {{ padding: 13px 14px; color: #1b1b1b; font-size: 14px; font-weight: 600; word-break: break-word; }}
            .footer {{ background: #f8fcfa; border-top: 1px solid #e4eee8; padding: 18px 32px; text-align: center; color: #6a766f; font-size: 12px; line-height: 1.5; }}
        </style>
    </head>
    <body>
        <div class="wrap">
            <div class="card">
                <div class="header">
                    <p class="brand">MindScope</p>
                    <p class="subtitle">Depression Screening Support</p>
                </div>
                <div class="body">
                    <h1>{_html_escape(title)}</h1>
                    <p class="intro">{_html_escape(intro)}</p>
                    {table_html}
                </div>
                <div class="footer">{footer_text}</div>
            </div>
        </div>
    </body>
    </html>
    """


def _plain_from_rows(title: str, intro: str, rows: list[tuple[str, object]] | None = None) -> str:
    lines = [title, "", intro, ""]
    for label, value in rows or []:
        lines.append(f"{label}: {value if value not in (None, '') else 'Not provided'}")
    return "\n".join(lines).strip()


async def send_email_async(
    to_email: str,
    subject: str,
    text_body: str,
    html_body: str | None = None,
) -> bool:
    """Send a generic transactional email using the configured SMTP account."""
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        logger.warning(
            "[EMAIL] SMTP not configured — transactional email skipped. "
            "Set SMTP_USER and SMTP_PASSWORD in .env"
        )
        logger.info(f"[EMAIL-DEV] To: {to_email} | Subject: {subject}\n{text_body}")
        return True

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"MindScope <{settings.SMTP_USER}>"
    msg["To"] = to_email
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body or _build_card_html(subject, text_body), "html"))

    return await asyncio.to_thread(_send_smtp, to_email, msg)


async def send_card_email_async(
    to_email: str,
    subject: str,
    title: str,
    intro: str,
    rows: list[tuple[str, object]] | None = None,
    footer: str = "",
) -> bool:
    return await send_email_async(
        to_email=to_email,
        subject=subject,
        text_body=_plain_from_rows(title, intro, rows),
        html_body=_build_card_html(title, intro, rows, footer),
    )


def send_otp_email(to_email: str, otp: str, user_name: str = "User") -> bool:
    """Send an OTP verification email (synchronous version for backward compat).

    Args:
        to_email: Recipient email address
        otp: The OTP code to send
        user_name: User's display name

    Returns:
        True if email was sent successfully, False otherwise
    """
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        logger.warning(
            "[EMAIL] SMTP not configured — OTP email skipped. "
            "Set SMTP_USER and SMTP_PASSWORD in .env"
        )
        logger.info(f"[EMAIL-DEV] OTP for {to_email}: {otp}")
        return True

    msg = _build_otp_message(to_email, otp, user_name)
    return _send_smtp(to_email, msg)


async def send_otp_email_async(to_email: str, otp: str, user_name: str = "User") -> bool:
    """Send an OTP verification email without blocking the event loop.

    Uses asyncio.to_thread() to run SMTP in a thread pool.
    """
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        logger.warning(
            "[EMAIL] SMTP not configured — OTP email skipped. "
            "Set SMTP_USER and SMTP_PASSWORD in .env"
        )
        logger.info(f"[EMAIL-DEV] OTP for {to_email}: {otp}")
        return True

    msg = _build_otp_message(to_email, otp, user_name)
    return await asyncio.to_thread(_send_smtp, to_email, msg)


def _build_otp_message(to_email: str, otp: str, user_name: str) -> MIMEMultipart:
    """Build the OTP email message."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"MindScope — Your Verification Code: {otp}"
    msg["From"] = f"MindScope <{settings.SMTP_USER}>"
    msg["To"] = to_email

    text_body = (
        f"Hi {user_name},\n\n"
        f"Your MindScope verification code is: {otp}\n\n"
        f"This code expires in {settings.OTP_EXPIRE_MINUTES} minutes.\n\n"
        f"If you did not request this, ignore this email."
    )

    html_body = _build_card_html(
        title=f"Hi {user_name}, verify your email",
        intro="Use the verification code below to complete your account setup.",
        rows=[
            ("Verification code", otp),
            ("Expires in", f"{settings.OTP_EXPIRE_MINUTES} minutes"),
        ],
        footer="If you did not request this code, you can safely ignore this email.",
    )

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    return msg
