"""Retry-политика для transient LLM/API ошибок.

Защищает рантайм от временных сбоев сети или API-провайдера 
(таймауты, лимиты запросов, 50x ошибки сервера).
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Final

import httpx
from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.services.swarm_agent.config import Settings

# Множество заморожено для O(1) поиска на скорости C-движка
_TRANSIENT_STATUS_CODES: Final[frozenset[int]] = frozenset(
    {408, 409, 425, 429, 500, 502, 503, 504}
)

# Неизменяемый кортеж строковых маркеров
_TRANSIENT_MARKERS: Final[tuple[str, ...]] = (
    "timeout",
    "ratelimit",
    "rate_limit",
    "connection",
    "serviceunavailable",
    "internalserver",
    "temporar",
)


def _status_code(exc: BaseException) -> int | None:
    """Безопасно извлекает HTTP status из глубин различных SDK-исключений."""
    
    # Прямой атрибут исключения
    if isinstance(code := getattr(exc, "status_code", None), int):
        return code
        
    # Вложенный атрибут объекта response
    if (resp := getattr(exc, "response", None)) and isinstance(
        rcode := getattr(resp, "status_code", None),
        int,
    ):
        return rcode
            
    return None


def is_transient_exception(exc: BaseException) -> bool:
    """Определяет временный сетевой/API сбой для запуска retry-цикла."""
    
    if isinstance(
        exc, 
        (
            httpx.TimeoutException, 
            httpx.NetworkError, 
            httpx.RemoteProtocolError
        )
    ):
        return True
        
    if _status_code(exc) in _TRANSIENT_STATUS_CODES:
        return True
        
    name = exc.__class__.__name__.lower()
    return any(marker in name for marker in _TRANSIENT_MARKERS)


async def ainvoke_with_retries(
    runnable: Any,
    messages: list[Any],
    config: RunnableConfig,
    *,
    settings: Settings,
    node_name: str,
) -> tuple[Any, int]:
    """Выполняет вызов с bounded exponential backoff и джиттером (jitter)."""
    
    max_retries = settings.node_max_retries
    retries = 0
    
    for attempt in range(max_retries + 1):
        try:
            # Оптимистичный вызов: отрабатывает мгновенно в большинстве случаев
            res = await runnable.ainvoke(messages, config)
            return res, retries
            
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries or not is_transient_exception(exc):
                raise
                
            retries += 1
            
            # Расчет экспоненциальной задержки
            base_delay = settings.retry_initial_delay_s * (2 ** attempt)
            delay = min(settings.retry_max_delay_s, base_delay)
            
            # Добавление джиттера для предотвращения Thundering Herd Problem
            delay += random.uniform(0.0, 0.25)
            
            logger.bind(node=node_name, retry=retries).warning(
                "Transient LLM error; retrying in {:.2f}s: {}",
                delay,
                exc.__class__.__name__,
            )
            
            await asyncio.sleep(delay)
            
    raise RuntimeError("unreachable retry state")
