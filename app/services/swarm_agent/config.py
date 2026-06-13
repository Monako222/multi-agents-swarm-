"""Локальные настройки swarm runtime.

Значения держим прямо в файле, без выноса структуры проекта в отдельный
конфиг-слой. Переменные окружения используются только для секретов и
операторских переопределений первого запуска.
"""

from __future__ import annotations

from functools import lru_cache
from os import getenv
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr


def _bool_env(name: str, default: bool) -> bool:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        if raw := getenv(name):
            return raw.strip()
    return default


def _secret_env(*names: str) -> SecretStr | None:
    raw = _env(*names)
    return SecretStr(raw) if raw else None


def _bool_env_any(*names: str, default: bool) -> bool:
    for name in names:
        raw = getenv(name)
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _int_env(*names: str, default: int) -> int:
    raw = _env(*names)
    return int(raw) if raw else default


def _float_env(*names: str, default: float) -> float:
    raw = _env(*names)
    return float(raw) if raw else default


class Settings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    OPENROUTER_API_KEY: SecretStr | None = Field(
        default=""
    )
    openrouter_base_url: str = Field(
        default_factory=lambda: _env(
            "OPENROUTER_BASE_URL",
            "SWARM_OPENROUTER_BASE_URL",
            default="https://openrouter.ai/api/v1",
        )
        or "https://openrouter.ai/api/v1"
    )
    openrouter_http_referer: str = Field(
        default_factory=lambda: _env(
            "OPENROUTER_HTTP_REFERER",
            "SWARM_OPENROUTER_HTTP_REFERER",
            default="https://agent",
        )
        or "https://agent"
    )
    openrouter_app_title: str = Field(
        default_factory=lambda: _env(
            "OPENROUTER_APP_TITLE",
            "SWARM_OPENROUTER_APP_TITLE",
            default="Swarm Agent",
        )
        or "Swarm Agent"
    )
    openrouter_use_proxy: bool = Field(
        default_factory=lambda: _bool_env_any(
            "OPENROUTER_USE_PROXY",
            "SWARM_OPENROUTER_USE_PROXY",
            default=False,
        )
    )
    openrouter_timeout_s: float = Field(
        default_factory=lambda: _float_env(
            "OPENROUTER_TIMEOUT_S",
            "SWARM_OPENROUTER_TIMEOUT_S",
            default=120.0,
        )
    )
    openrouter_connect_timeout_s: float = Field(
        default_factory=lambda: _float_env(
            "OPENROUTER_CONNECT_TIMEOUT_S",
            "SWARM_OPENROUTER_CONNECT_TIMEOUT_S",
            default=15.0,
        )
    )
    openrouter_pool_timeout_s: float = Field(
        default_factory=lambda: _float_env(
            "OPENROUTER_POOL_TIMEOUT_S",
            "SWARM_OPENROUTER_POOL_TIMEOUT_S",
            default=15.0,
        )
    )
    openrouter_max_retries: int = Field(
        default_factory=lambda: _int_env(
            "OPENROUTER_MAX_RETRIES",
            "SWARM_OPENROUTER_MAX_RETRIES",
            default=3,
        )
    )
    proxy_url: str | None = Field(
        default_factory=lambda: getenv("HTTPS_PROXY") or getenv("HTTP_PROXY")
    )

    debug: bool = Field(default_factory=lambda: _bool_env("SWARM_DEBUG", False))
    default_model_alias: str = "fast"
    allow_parallel_tool_calls: bool | None = True
    recover_plain_text_answer: bool = True

    max_total_steps: int = 64
    max_agent_loops: int = 8
    max_tool_calls_per_turn: int = 8
    max_parallel_tool_calls: int = 4

    recent_messages_limit: int = 12
    memory_summary_char_limit: int = 8_000
    active_message_char_limit: int = 40_000
    state_part_char_limit: int = 4_000
    data_part_char_limit: int = 8_000
    file_part_char_limit: int = 6_000
    runtime_context_char_limit: int = 24_000

    node_max_retries: int = 2
    retry_initial_delay_s: float = 0.5
    retry_max_delay_s: float = 8.0

    web_search_model_id: str = Field(
        default_factory=lambda: _env(
            "WEB_SEARCH_MODEL_ID",
            "SWARM_WEB_SEARCH_MODEL_ID",
            default="google/gemini-2.5-flash-lite",
        )
        or "google/gemini-2.5-flash-lite"
    )
    web_search_engine: str = Field(
        default_factory=lambda: _env(
            "WEB_SEARCH_ENGINE",
            "SWARM_WEB_SEARCH_ENGINE",
            default="auto",
        )
        or "auto"
    )
    web_search_max_queries: int = Field(
        default_factory=lambda: _int_env(
            "WEB_SEARCH_MAX_QUERIES",
            "SWARM_WEB_SEARCH_MAX_QUERIES",
            default=5,
        )
    )
    web_search_max_results: int = Field(
        default_factory=lambda: _int_env(
            "WEB_SEARCH_MAX_RESULTS",
            "SWARM_WEB_SEARCH_MAX_RESULTS",
            default=5,
        )
    )
    web_search_max_domains: int = Field(
        default_factory=lambda: _int_env(
            "WEB_SEARCH_MAX_DOMAINS",
            "SWARM_WEB_SEARCH_MAX_DOMAINS",
            default=10,
        )
    )
    web_search_context_size: str = Field(
        default_factory=lambda: _env(
            "WEB_SEARCH_CONTEXT_SIZE",
            "SWARM_WEB_SEARCH_CONTEXT_SIZE",
            default="medium",
        )
        or "medium"
    )

    user_retry_later_message: str = (
        "Сейчас не получилось обработать запрос. Попробуйте, пожалуйста, чуть позже."
    )
    user_empty_query_message: str = "Пожалуйста, отправьте непустой запрос."
    user_invalid_input_message: str = (
        "Не получилось обработать входные данные. Проверьте запрос и файлы и попробуйте снова."
    )

    def model_post_init(self, __context: Any) -> None:
        positive_ints = (
            "max_total_steps",
            "max_agent_loops",
            "max_tool_calls_per_turn",
            "max_parallel_tool_calls",
            "recent_messages_limit",
            "active_message_char_limit",
            "web_search_max_queries",
            "web_search_max_results",
            "web_search_max_domains",
            "openrouter_max_retries",
        )
        for name in positive_ints:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")

        positive_floats = (
            "openrouter_timeout_s",
            "openrouter_connect_timeout_s",
            "openrouter_pool_timeout_s",
        )
        for name in positive_floats:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

        web_search_engine = self.web_search_engine.strip().lower()
        web_search_context_size = self.web_search_context_size.strip().lower()
        object.__setattr__(self, "web_search_engine", web_search_engine)
        object.__setattr__(self, "web_search_context_size", web_search_context_size)

        if web_search_engine not in {
            "auto",
            "native",
            "exa",
            "firecrawl",
            "parallel",
            "perplexity",
        }:
            raise ValueError("web_search_engine has unsupported value")

        if web_search_context_size not in {"low", "medium", "high"}:
            raise ValueError("web_search_context_size has unsupported value")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
