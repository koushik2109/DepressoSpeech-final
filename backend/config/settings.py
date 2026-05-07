"""Application settings loaded from environment / .env file."""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"

    # Database – SQLite for simplicity
    DATABASE_URL: str = f"sqlite+aiosqlite:///{(Path(__file__).resolve().parent.parent / 'mindscope.db').as_posix()}"
    DATABASE_ECHO: bool = False

    # JWT
    JWT_SECRET_KEY: str = "CHANGE_ME_TO_A_RANDOM_64_CHAR_STRING"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    # Storage
    STORAGE_LOCAL_PATH: str = "./storage/audio"
    AUDIO_MAX_FILE_SIZE_MB: int = 100
    AUDIO_ALLOWED_EXTENSIONS: str = ".wav,.mp3,.flac,.ogg,.m4a,.webm"
    VIDEO_ALLOWED_EXTENSIONS: str = ".mp4,.avi,.mov,.mkv,.webm"
    VIDEO_MAX_FILE_SIZE_MB: int = 500
    MULTIMODAL_STORAGE_PATH: str = "./storage/multimodal"

    # ML
    ML_MODEL_PATH: str = "../Model/checkpoints/best_model.pt"
    ML_CONFIG_PATH: str = "../Model/configs/inference_config.yaml"
    ML_DEVICE: str = "auto"
    ML_MODEL_URL: str = "http://localhost:8001"

    # Admin defaults
    ADMIN_DEFAULT_EMAIL: str = "admin@mindscope.ai"
    ADMIN_DEFAULT_PASSWORD: str = "Admin@2026!"

    # SMTP (for OTP emails)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    OTP_EXPIRE_MINUTES: int = 10

    # Google OAuth
    GOOGLE_CLIENT_ID: str = ""

    @property
    def allowed_extensions_set(self) -> set:
        return set(self.AUDIO_ALLOWED_EXTENSIONS.split(","))


@lru_cache()
def get_settings() -> Settings:
    return Settings()
