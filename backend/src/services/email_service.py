"""Email service for MindScope.

Delivery strategy (tried in order):
  1. Resend HTTP API  — works on Render Free Tier (port 443, never blocked)
  2. smtplib fallback — works on local / VPS where port 465/587 is open

Root cause of Render SMTP failure:
  Render Free Tier firewalls block outbound TCP on ports 25, 465, and 587 at
  the hypervisor level.  The OS raises `Errno 101 - Network is unreachable`
  *before* the TLS handshake because the SYN packet is dropped — not because
  of bad credentials or wrong config.  Switching to an HTTP-based provider
  (Resend, SendGrid, Mailgun) bypasses this entirely since they use port 443.
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


# ── OTP helpers ───────────────────────────────────────────────────────────────

def generate_otp(length: int = 6) -> str:
    """Generate a random numeric OTP."""
    return "".join(random.choices(string.digits, k=length))


# ── HTML template ─────────────────────────────────────────────────────────────

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


# ── Resend HTTP delivery (primary — works on Render Free Tier) ────────────────

def _send_via_resend(to_email: str, subject: str, html_body: str, text_body: str) -> bool:
    """Send email via Resend HTTP API (port 443 — never blocked by any host).

    Render Free Tier blocks outbound SMTP ports (25, 465, 587) at the firewall.
    Resend uses a simple HTTPS REST call, which is always allowed.

    Sign up free at https://resend.com — 3,000 emails/month on free tier.
    """
    try:
        import resend  # pip install resend
        resend.api_key = settings.RESEND_API_KEY

        params: resend.Emails.SendParams = {
            "from": settings.RESEND_FROM_EMAIL,
            "to": [to_email],
            "subject": subject,
            "html": html_body,
            "text": text_body,
        }
        response = resend.Emails.send(params)
        email_id = response.get("id", "unknown") if isinstance(response, dict) else getattr(response, "id", "unknown")
        logger.info("[EMAIL] Resend delivery OK → id=%s to=%s", email_id, to_email)
        return True
    except Exception as exc:
        logger.error("[EMAIL] Resend delivery failed: %s", exc)
        return False


# ── SMTP fallback (local dev / VPS where port 465/587 is open) ────────────────

def _send_via_smtp(to_email: str, msg: MIMEMultipart) -> bool:
    """SMTP send — works locally but blocked on Render Free Tier.

    Errno 101 (Network is unreachable) means the outbound TCP SYN packet was
    dropped before reaching smtp.gmail.com — the connection never starts.
    This is a host-level firewall rule, not a credentials or config issue.
    """
    port = settings.SMTP_PORT or 465
    try:
        if port == 465:
            with smtplib.SMTP_SSL(settings.SMTP_HOST, port, timeout=10) as server:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(settings.SMTP_USER, to_email, msg.as_string())
        else:
            with smtplib.SMTP(settings.SMTP_HOST, port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(settings.SMTP_USER, to_email, msg.as_string())
        logger.info("[EMAIL] SMTP delivery OK to %s", to_email)
        return True
    except OSError as exc:
        if exc.errno == 101:
            logger.error(
                "[EMAIL] SMTP blocked (Errno 101 - Network unreachable). "
                "Render Free Tier blocks outbound SMTP ports. "
                "Set RESEND_API_KEY in your Render environment variables."
            )
        else:
            logger.error("[EMAIL] SMTP OSError to %s: %s", to_email, exc)
        return False
    except Exception as exc:
        logger.error("[EMAIL] SMTP failed to %s: %s", to_email, exc)
        return False


# ── Smart dispatcher — tries Resend first, falls back to SMTP ─────────────────

def _dispatch_email(to_email: str, subject: str, html_body: str, text_body: str) -> bool:
    """Try Resend first (works everywhere), fall back to SMTP (local only)."""

    # 1. Try Resend if API key is configured
    if settings.RESEND_API_KEY:
        return _send_via_resend(to_email, subject, html_body, text_body)

    # 2. Try SMTP if credentials are configured (works locally, blocked on Render)
    if settings.SMTP_USER and settings.SMTP_PASSWORD:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"MindScope <{settings.SMTP_USER}>"
        msg["To"] = to_email
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        return _send_via_smtp(to_email, msg)

    # 3. No provider configured — log and skip
    logger.warning(
        "[EMAIL] No email provider configured. "
        "Set RESEND_API_KEY (recommended) or SMTP_USER + SMTP_PASSWORD in environment. "
        "Email to %s skipped.",
        to_email,
    )
    logger.info("[EMAIL-DEV] Subject: %s\n%s", subject, text_body)
    return True  # Non-fatal in dev


# ── Public async API ──────────────────────────────────────────────────────────

async def send_email_async(
    to_email: str,
    subject: str,
    text_body: str,
    html_body: str | None = None,
) -> bool:
    """Send a transactional email. Uses Resend if configured, else SMTP."""
    _html = html_body or _build_card_html(subject, text_body)
    return await asyncio.to_thread(_dispatch_email, to_email, subject, _html, text_body)


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


async def send_otp_email_async(to_email: str, otp: str, user_name: str = "User") -> bool:
    """Send OTP verification email without blocking the event loop."""
    msg = _build_otp_message(to_email, otp, user_name)
    subject = f"MindScope — Your Verification Code: {otp}"
    html_body = _build_card_html(
        title=f"Hi {user_name}, verify your email",
        intro="Use the verification code below to complete your account setup.",
        rows=[
            ("Verification code", otp),
            ("Expires in", f"{settings.OTP_EXPIRE_MINUTES} minutes"),
        ],
        footer="If you did not request this code, you can safely ignore this email.",
    )
    text_body = (
        f"Hi {user_name},\n\n"
        f"Your MindScope verification code is: {otp}\n\n"
        f"This code expires in {settings.OTP_EXPIRE_MINUTES} minutes.\n\n"
        f"If you did not request this, ignore this email."
    )
    return await asyncio.to_thread(_dispatch_email, to_email, subject, html_body, text_body)


def send_otp_email(to_email: str, otp: str, user_name: str = "User") -> bool:
    """Synchronous OTP send (backward-compat wrapper)."""
    subject = f"MindScope — Your Verification Code: {otp}"
    html_body = _build_card_html(
        title=f"Hi {user_name}, verify your email",
        intro="Use the verification code below to complete your account setup.",
        rows=[
            ("Verification code", otp),
            ("Expires in", f"{settings.OTP_EXPIRE_MINUTES} minutes"),
        ],
        footer="If you did not request this code, you can safely ignore this email.",
    )
    text_body = (
        f"Hi {user_name},\n\n"
        f"Your MindScope verification code is: {otp}\n\n"
        f"This code expires in {settings.OTP_EXPIRE_MINUTES} minutes.\n\n"
        f"If you did not request this, ignore this email."
    )
    return _dispatch_email(to_email, subject, html_body, text_body)


def _build_otp_message(to_email: str, otp: str, user_name: str) -> MIMEMultipart:
    """Build the OTP MIME message (used by SMTP fallback path)."""
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
