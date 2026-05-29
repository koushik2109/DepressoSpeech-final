"""Authentication routes: register, login, admin login, logout, me, OTP verification."""

import logging
import secrets
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

# pyrefly: ignore [missing-import]
from database import get_db
# pyrefly: ignore [missing-import]
from src.models import Doctor, User
# pyrefly: ignore [missing-import]
from src.utils.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
)
# pyrefly: ignore [missing-import]
from src.middleware.deps import get_current_user
# pyrefly: ignore [missing-import]
from src.services.email_service import generate_otp, send_otp_email_async
# pyrefly: ignore [missing-import]
from config.settings import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()
logger = logging.getLogger("mindscope.auth")

# ── OTP rate limiting (in-memory, resets on restart) ───
_otp_attempts: dict = defaultdict(list)  # email -> [timestamps]
MAX_OTP_ATTEMPTS = 5
OTP_WINDOW_SECONDS = 600  # 10 minutes


def _check_otp_rate_limit(email: str) -> None:
    """Raise 429 if too many OTP attempts for this email."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=OTP_WINDOW_SECONDS)
    _otp_attempts[email] = [t for t in _otp_attempts[email] if t > cutoff]
    if len(_otp_attempts[email]) >= MAX_OTP_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many attempts. Please wait {OTP_WINDOW_SECONDS // 60} minutes.",
        )
    _otp_attempts[email].append(now)


# ── Schemas ────────────────────────────────────────────

class RegisterRequest(BaseModel):
    role: str = Field(..., pattern="^(patient|doctor)$")
    name: str = Field(..., min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(..., min_length=6)
    # Patient fields
    age: Optional[int] = None
    basicInfo: Optional[str] = None
    # Doctor fields
    specialization: Optional[str] = None
    licenseNumber: Optional[str] = None
    clinicName: Optional[str] = None
    yearsExperience: Optional[int] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AdminLoginRequest(BaseModel):
    adminId: str
    password: str


class LogoutRequest(BaseModel):
    refreshToken: Optional[str] = None


class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)


class ResendOtpRequest(BaseModel):
    email: EmailStr


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)
    newPassword: str = Field(..., min_length=6)


class GoogleAuthRequest(BaseModel):
    credential: str = Field(..., min_length=1)


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    age: Optional[int] = Field(default=None, ge=0, le=120)
    basicInfo: Optional[str] = Field(default=None, max_length=2000)


# ── Register ───────────────────────────────────────────

@router.post("/register", status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check duplicate email
    existing = await db.execute(select(User).where(User.email == body.email.lower()))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    # Generate OTP
    otp = generate_otp()
    otp_expires = datetime.now(timezone.utc) + timedelta(minutes=settings.OTP_EXPIRE_MINUTES)

    user = User(
        role=body.role,
        name=body.name,
        email=body.email.lower(),
        password_hash=hash_password(body.password),
        age=body.age,
        basic_info=body.basicInfo,
        specialization=body.specialization,
        license_number=body.licenseNumber,
        clinic_name=body.clinicName,
        years_experience=body.yearsExperience,
        is_verified=False,
        verification_otp=otp,
        otp_expires_at=otp_expires,
    )
    db.add(user)
    await db.flush()

    if user.role == "doctor":
        db.add(
            Doctor(
                user_id=user.id,
                name=user.name,
                email=user.email,
                fee=100.0,
                is_available=False,
            )
        )
        await db.flush()

    # Send OTP email asynchronously (non-blocking)
    await send_otp_email_async(to_email=user.email, otp=otp, user_name=user.name)

    return {
        "user": _user_response(user),
        "requiresVerification": True,
        "message": "Account created. Please verify your email with the OTP sent.",
    }


# ── Verify OTP ─────────────────────────────────────────

@router.post("/verify-otp")
async def verify_otp(body: VerifyOtpRequest, db: AsyncSession = Depends(get_db)):
    _check_otp_rate_limit(body.email.lower())

    result = await db.execute(select(User).where(User.email == body.email.lower()))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="No account found with this email.")

    if user.is_verified:
        return {"success": True, "message": "Email is already verified."}

    if not user.verification_otp:
        raise HTTPException(status_code=400, detail="No OTP has been generated. Please request a new one.")

    if user.otp_expires_at:
        expire_dt = user.otp_expires_at
        now_dt = datetime.now(timezone.utc)
        # asyncpg (PostgreSQL) returns tz-aware datetimes; normalize if naive
        if expire_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=None)
        if now_dt > expire_dt:
            raise HTTPException(status_code=410, detail="OTP has expired. Please request a new one.")

    if user.verification_otp != body.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP. Please check and try again.")

    # Mark verified, clear OTP
    user.is_verified = True
    user.verification_otp = None
    user.otp_expires_at = None
    await db.flush()

    return {
        "success": True,
        "message": "Email verified successfully!",
        "user": _user_response(user),
    }


# ── Resend OTP ─────────────────────────────────────────

@router.post("/resend-otp")
async def resend_otp(body: ResendOtpRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email.lower()))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="No account found with this email.")

    if user.is_verified:
        return {"success": True, "message": "Email is already verified."}

    # Generate new OTP
    otp = generate_otp()
    user.verification_otp = otp
    user.otp_expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.OTP_EXPIRE_MINUTES)
    await db.flush()

    sent = await send_otp_email_async(to_email=user.email, otp=otp, user_name=user.name)
    if not sent:
        # Email failed (SMTP blocked on this network) — log OTP to server console
        # so admins/devs can relay it manually. Never block the user.
        logger.warning(
            f"[OTP-FALLBACK] Email delivery failed for {user.email}. "
            f"OTP={otp} (expires in {settings.OTP_EXPIRE_MINUTES} min)"
        )

    return {
        "success": True,
        "message": "A new OTP has been sent to your email.",
    }


# ── Forgot Password ───────────────────────────────────

@router.post("/forgot-password")
async def forgot_password(body: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email.lower()))
    user = result.scalar_one_or_none()

    if not user:
        # Don't reveal whether the email exists
        return {"success": True, "message": "If that email is registered, an OTP has been sent."}

    otp = generate_otp()
    user.verification_otp = otp
    user.otp_expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.OTP_EXPIRE_MINUTES)
    await db.flush()

    await send_otp_email_async(to_email=user.email, otp=otp, user_name=user.name)

    return {"success": True, "message": "If that email is registered, an OTP has been sent."}


@router.post("/reset-password")
async def reset_password(body: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email.lower()))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="No account found with this email.")

    if not user.verification_otp:
        raise HTTPException(status_code=400, detail="No OTP has been generated. Please request one first.")

    if user.otp_expires_at:
        expire_dt = user.otp_expires_at
        now_dt = datetime.now(timezone.utc)
        if expire_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=None)
        if now_dt > expire_dt:
            raise HTTPException(status_code=410, detail="OTP has expired. Please request a new one.")

    if user.verification_otp != body.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP. Please check and try again.")

    user.password_hash = hash_password(body.newPassword)
    user.verification_otp = None
    user.otp_expires_at = None
    await db.flush()

    return {"success": True, "message": "Password has been reset successfully."}


# ── Login ──────────────────────────────────────────────

@router.post("/login")
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email.lower()))
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if user.status != "active":
        raise HTTPException(status_code=423, detail="Account is locked.")

    # Block login until email is verified (Google OAuth users are pre-verified)
    if not user.is_verified:
        raise HTTPException(
            status_code=403,
            detail="Email not verified. Please check your inbox for the OTP.",
        )

    tokens = _issue_tokens(user)
    return {
        **tokens,
        "user": _user_response(user),
    }


# ── Admin Login ────────────────────────────────────────

@router.post("/admin/login")
async def admin_login(body: AdminLoginRequest, db: AsyncSession = Depends(get_db)):
    # Guard: secrets.compare_digest requires non-None strings
    if not settings.ADMIN_DEFAULT_EMAIL or not settings.ADMIN_DEFAULT_PASSWORD:
        raise HTTPException(status_code=503, detail="Admin credentials not configured.")

    # Check against default admin account first
    if secrets.compare_digest(body.adminId, settings.ADMIN_DEFAULT_EMAIL) and secrets.compare_digest(body.password, settings.ADMIN_DEFAULT_PASSWORD):
        # Find or create admin user
        result = await db.execute(select(User).where(User.email == settings.ADMIN_DEFAULT_EMAIL))
        admin = result.scalar_one_or_none()

        if not admin:
            admin = User(
                role="admin",
                name="Admin",
                email=settings.ADMIN_DEFAULT_EMAIL,
                password_hash=hash_password(settings.ADMIN_DEFAULT_PASSWORD),
                is_verified=True,
            )
            db.add(admin)
            await db.flush()

        tokens = _issue_tokens(admin)
        return {
            **tokens,
            "admin": {"id": admin.id, "adminId": admin.email},
        }

    raise HTTPException(status_code=401, detail="Invalid admin credentials.")


# ── Google OAuth ──────────────────────────────────────

@router.post("/google")
async def google_login(body: GoogleAuthRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate with a Google ID token (from Google Sign-In)."""
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=501, detail="Google OAuth is not configured on this server.")

    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        idinfo = google_id_token.verify_oauth2_token(
            body.credential,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID,
        )
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Google token.")

    email = idinfo.get("email", "").lower()
    name = idinfo.get("name", email.split("@")[0])

    if not email:
        raise HTTPException(status_code=400, detail="Google account has no email.")

    # Find or create user
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user:
        user = User(
            role="patient",
            name=name,
            email=email,
            password_hash=hash_password(idinfo["sub"]),  # random-ish placeholder
            is_verified=True,  # Google accounts are pre-verified
        )
        db.add(user)
        await db.flush()

    if user.status != "active":
        raise HTTPException(status_code=423, detail="Account is locked.")

    # Auto-verify if they signed up via email but hadn't verified yet
    if not user.is_verified:
        user.is_verified = True
        user.verification_otp = None
        user.otp_expires_at = None
        await db.flush()

    tokens = _issue_tokens(user)
    return {
        **tokens,
        "user": _user_response(user),
    }


# ── Logout ─────────────────────────────────────────────

@router.post("/logout")
async def logout(body: LogoutRequest):
    return {"success": True}


# ── Me ─────────────────────────────────────────────────

@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return {"user": _user_response(user)}


@router.put("/me")
async def update_me(
    body: UpdateProfileRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if user.role != "patient":
        raise HTTPException(status_code=403, detail="Patient profile update required.")

    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name is required.")
        user.name = name
    if body.age is not None:
        user.age = body.age
    if body.basicInfo is not None:
        user.basic_info = body.basicInfo.strip() or None

    await db.flush()
    return {"user": _user_response(user)}


# ── Helpers ────────────────────────────────────────────

def _issue_tokens(user: User) -> dict:
    payload = {"sub": user.id, "email": user.email, "role": user.role}
    return {
        "accessToken": create_access_token(payload),
        "refreshToken": create_refresh_token(payload),
        "expiresIn": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


def _user_response(user: User) -> dict:
    data = {
        "id": user.id,
        "role": user.role,
        "name": user.name,
        "email": user.email,
        "isVerified": user.is_verified if user.is_verified is not None else False,
        "createdAt": user.created_at.isoformat() if user.created_at else None,
    }
    if user.role == "patient":
        data["age"] = user.age
        data["basicInfo"] = user.basic_info
    elif user.role == "doctor":
        data["specialization"] = user.specialization
        data["licenseNumber"] = user.license_number
        data["clinicName"] = user.clinic_name
        data["yearsExperience"] = user.years_experience
    return data
