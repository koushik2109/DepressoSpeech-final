"""
Provision new user account: gsbkoushik@gmail.com
Run from the /backend directory:
    python scripts/provision_user.py
"""

import asyncio
import sys
import os

# Ensure the backend root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from database.base import async_session_factory, init_db
from src.models import User
from src.utils.auth import hash_password
from config.settings import get_settings

settings = get_settings()

NEW_USER = {
    "role": "patient",
    "name": "Koushik GSB",
    "email": "gsbkoushik@gmail.com",
    "password": "Koushik_21092007",
    "age": None,
    "basic_info": None,
}


async def provision():
    print(f"[provision] Connecting to: {settings.DATABASE_URL[:60]}...")
    await init_db()

    async with async_session_factory() as session:
        # Check if user already exists
        result = await session.execute(
            select(User).where(User.email == NEW_USER["email"].lower())
        )
        existing = result.scalar_one_or_none()

        if existing:
            print(f"[provision] User {NEW_USER['email']} already exists (id={existing.id}). Skipping.")
            return

        user = User(
            role=NEW_USER["role"],
            name=NEW_USER["name"],
            email=NEW_USER["email"].lower(),
            password_hash=hash_password(NEW_USER["password"]),
            age=NEW_USER.get("age"),
            basic_info=NEW_USER.get("basic_info"),
            is_verified=True,  # Pre-verified — no OTP required
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        print(f"[provision] ✓ Created user: {user.email} (id={user.id}, role={user.role})")


if __name__ == "__main__":
    asyncio.run(provision())
