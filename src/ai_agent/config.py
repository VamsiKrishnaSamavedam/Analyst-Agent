"""Centralized, validated configuration for the application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    gemini_api_key: str | None
    llm_thinking_level: Literal["low", "medium", "high"]
    db_host: str | None
    db_port: int
    db_name: str | None
    db_user: str | None
    db_password: str | None
    db_write_host: str | None
    db_write_port: int
    db_write_name: str | None
    db_write_user: str | None
    db_write_password: str | None

    @property
    def database_is_configured(self) -> bool:
        return all((self.db_host, self.db_name, self.db_user, self.db_password))

    def database_config(self) -> dict[str, object]:
        if not self.database_is_configured:
            raise RuntimeError(
                "Database credentials are incomplete. Copy .env.example to .env and "
                "set DB_HOST, DB_NAME, DB_USER, and DB_PASSWORD."
            )
        return {
            "host": self.db_host,
            "port": self.db_port,
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
        }

    @property
    def write_database_is_configured(self) -> bool:
        return all((self.db_write_host, self.db_write_name, self.db_write_user, self.db_write_password))

    def write_database_config(self) -> dict[str, object]:
        if not self.write_database_is_configured:
            raise RuntimeError(
                "PostgreSQL writer credentials are incomplete. Set DB_WRITE_USER and DB_WRITE_PASSWORD before staging data."
            )
        return {
            "host": self.db_write_host,
            "port": self.db_write_port,
            "dbname": self.db_write_name,
            "user": self.db_write_user,
            "password": self.db_write_password,
        }


def get_settings() -> Settings:
    """Load local development variables without ever printing secret values."""

    load_dotenv()
    raw_port = os.getenv("DB_PORT", os.getenv("port", "5432"))
    try:
        port = int(raw_port)
    except ValueError as error:
        raise RuntimeError("DB_PORT must be a valid integer.") from error

    raw_write_port = os.getenv("DB_WRITE_PORT", raw_port)
    try:
        write_port = int(raw_write_port)
    except ValueError as error:
        raise RuntimeError("DB_WRITE_PORT must be a valid integer.") from error

    thinking_level = os.getenv("LLM_THINKING_LEVEL", "medium").lower()
    if thinking_level not in {"low", "medium", "high"}:
        raise RuntimeError("LLM_THINKING_LEVEL must be low, medium, or high.")

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        llm_thinking_level=thinking_level,  # type: ignore[arg-type]
        db_host=os.getenv("DB_HOST", os.getenv("host")),
        db_port=port,
        db_name=os.getenv("DB_NAME", os.getenv("database", os.getenv("dbname"))),
        db_user=os.getenv("DB_USER", os.getenv("user")),
        db_password=os.getenv("DB_PASSWORD", os.getenv("password")),
        # Local prototype compatibility: existing projects use one working
        # PostgreSQL login via legacy lowercase keys. Production deployments
        # can and should override these with a separate DB_WRITE_* account.
        db_write_host=os.getenv("DB_WRITE_HOST", os.getenv("DB_HOST", os.getenv("host"))),
        db_write_port=write_port,
        db_write_name=os.getenv("DB_WRITE_NAME", os.getenv("DB_NAME", os.getenv("database", os.getenv("dbname")))),
        db_write_user=os.getenv("DB_WRITE_USER", os.getenv("DB_USER", os.getenv("user"))),
        db_write_password=os.getenv("DB_WRITE_PASSWORD", os.getenv("DB_PASSWORD", os.getenv("password"))),
    )
