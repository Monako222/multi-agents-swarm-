"""Мелкие runtime-утилиты без привязки к LangGraph.

Вынесены повторяющиеся операции: чтение роли/контента сообщения,
нормализация списков, prompt/state budget, безопасная обработка
исключений и слияние нескольких tool-update в единый поток.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import orjson
from pydantic import BaseModel

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


def clip(text: str, limit: int, *, suffix: str = " ...[truncated]") -> str:
    """Обрезать строку с жёстким лимитом символов."""
    if limit <= 0 or len(text) <= limit:
        return text
    cut = max(0, limit - len(suffix))
    return text[:cut].rstrip() + suffix


def clip_tail(text: str, limit: int, *, prefix: str = "...[earlier memory truncated]\n") -> str:
    """Оставить свежий хвост длинной памяти."""
    if limit <= 0 or len(text) <= limit:
        return text
    keep = max(0, limit - len(prefix))
    return prefix + text[-keep:].lstrip()


def to_json(value: Any) -> str:
    """Компактно сериализовать value для prompt-контекста."""
    if isinstance(value, BaseModel):
        return value.model_dump_json(exclude_unset=True, exclude_none=True)
    return orjson.dumps(value, default=str, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def state_part(title: str, value: Any, *, max_chars: int) -> str:
    """Сериализовать один канал state с ограничением размера."""
    if value in (None, "", [], {}, ()):
        return ""
    try:
        serialized = to_json(value)
    except Exception:  # noqa: BLE001 - сборка prompt не должна валить граф.
        serialized = repr(value)
    return f"[{title.upper()}]\n{clip(serialized, max_chars)}\n"


def uniq_lines(text: str, *, max_seen: int = 4_096) -> str:
    """Удалить повторяющиеся строки, сохраняя порядок и ограничивая память."""
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        key = line.strip()
        if key and key in seen:
            continue
        if key:
            if len(seen) >= max_seen:
                seen.clear()
            seen.add(key)
        out.append(line)
    return "\n".join(out)


def safe_error_text(exc: BaseException, *, limit: int = 1_500) -> str:
    """Сжимает трейсбэк ошибки для аккуратной записи в логи и snapshot."""
    
    text = str(exc).strip() or exc.__class__.__name__
    return clip(text, limit)


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
