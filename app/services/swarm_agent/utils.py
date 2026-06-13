"""Мелкие runtime-утилиты без привязки к LangGraph.

Вынесены повторяющиеся операции: чтение роли/контента сообщения,
нормализация списков, безопасная обработка исключений и слияние 
нескольких tool-update в единый поток.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from app.services.swarm_agent.text import truncate_text

# ---------------------------------------------------------------------------
# Константы для сверхбыстрого O(1) роутинга при слиянии стейтов
# ---------------------------------------------------------------------------
_LIST_KEYS = frozenset({
    "messages", 
    "errors", 
    "history", 
    "in_files", 
    "out_files"
})

_PATCH_KEYS = frozenset({
    "context", 
    "space", 
    "data", 
    "workspace"
})


def as_list(value: Any) -> list[Any]:
    """Приводит скаляры к списку, не разбивая строки и объекты на символы."""
    
    if value is None:
        return []
        
    if isinstance(value, list):
        return value
        
    if isinstance(value, tuple):
        return list(value)
        
    # Защита от распада строк, байтов, словарей и Pydantic-моделей
    if isinstance(value, str | bytes | bytearray | Mapping | BaseModel):
        return [value]
        
    if isinstance(value, Sequence):
        return list(value)
        
    return [value]


def clean_payload(**kwargs: Any) -> dict[str, Any]:
    """Оставляет только валидные значения, чтобы patch не затирал state."""
    
    return {k: v for k, v in kwargs.items() if v is not None}


def message_role(message: Any) -> str:
    """O(1) извлечение роли из LangChain/OpenAI-подобного сообщения."""
    
    return str(
        getattr(message, "type", None)
        or getattr(message, "role", None)
        or message.__class__.__name__
    ).lower()


def message_content(message: Any) -> str:
    """Безопасное извлечение текста, обходя мультимодальные блоки."""
    
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def safe_error_text(exc: BaseException, *, limit: int = 1_500) -> str:
    """Сжимает трейсбэк ошибки для аккуратной записи в логи и snapshot."""
    
    text = str(exc).strip() or exc.__class__.__name__
    return truncate_text(text, limit)


def _merge_patch(
    left: Mapping[str, Any], 
    right: Mapping[str, Any]
) -> dict[str, Any]:
    """Рекурсивное глубокое слияние словарей без потери списков."""
    
    merged = dict(left)
    
    for k, v in right.items():
        old = merged.get(k)
        
        if isinstance(old, Mapping) and isinstance(v, Mapping):
            merged[k] = _merge_patch(old, v)
            
        elif isinstance(old, list) or isinstance(v, list):
            merged[k] = [*as_list(old), *as_list(v)]
            
        else:
            merged[k] = v
            
    return merged


def _merge_metrics(
    left: Mapping[str, Any], 
    right: Mapping[str, Any]
) -> dict[str, Any]:
    """Суммирует числовые метрики из параллельных tool calls."""
    
    merged = dict(left)
    
    for k, v in right.items():
        if isinstance(v, int | float):
            merged[k] = int(merged.get(k, 0)) + int(v)
        else:
            merged[k] = v
            
    return merged


def merge_update(
    dst: dict[str, Any], 
    src: Mapping[str, Any]
) -> dict[str, Any]:
    """Объединяет параллельные Command.update в один LangGraph update.
    
    Редьюсеры увидят только итоговый стейт: списки склеиваются, метрики 
    суммируются, словари стейта глубоко сливаются, а скаляры берутся 
    из последнего routing tool.
    """
    
    for k, v in src.items():
        
        if v is None:
            continue
            
        if k in _LIST_KEYS:
            dst[k] = [*as_list(dst.get(k)), *as_list(v)]
            continue
            
        if k == "metrics" and isinstance(v, Mapping):
            base = dst.get(k)
            base_dict = base if isinstance(base, Mapping) else {}
            dst[k] = _merge_metrics(base_dict, v)
            continue
            
        if k in _PATCH_KEYS and isinstance(v, Mapping):
            base = dst.get(k)
            base_dict = base if isinstance(base, Mapping) else {}
            dst[k] = _merge_patch(base_dict, v)
            continue
            
        dst[k] = v
        
    return dst
