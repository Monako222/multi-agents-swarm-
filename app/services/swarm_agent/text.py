"""Безопасные текстовые и JSON-утилиты для prompt/state."""

from __future__ import annotations

from typing import Any

import orjson
from pydantic import BaseModel


def approx_token_count(text: str) -> int:
    """Грубая оценка токенов для English/Cyrillic текста.

    Используется только для диагностик/бюджетов, не для биллинга.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def truncate_text(text: str, limit: int, *, suffix: str = " ...[truncated]") -> str:
    """Обрезать строку с жёстким лимитом символов."""
    if limit <= 0 or len(text) <= limit:
        return text
    cut = max(0, limit - len(suffix))
    return text[:cut].rstrip() + suffix


def tail_text(text: str, limit: int, *, prefix: str = "...[earlier memory truncated]\n") -> str:
    """Оставить свежий хвост длинной памяти."""
    if limit <= 0 or len(text) <= limit:
        return text
    keep = max(0, limit - len(prefix))
    return prefix + text[-keep:].lstrip()


def dumps_compact(value: Any) -> str:
    """Компактно сериализовать value для prompt-контекста."""
    if isinstance(value, BaseModel):
        return value.model_dump_json(exclude_unset=True, exclude_none=True)
    return orjson.dumps(value, default=str, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def dump_state_part(title: str, value: Any, *, max_chars: int) -> str:
    """Сериализовать один канал state с ограничением размера."""
    if value in (None, "", [], {}, ()):  # быстрый путь для пустых каналов
        return ""
    try:
        serialized = dumps_compact(value)
    except Exception:  # noqa: BLE001 - сборка prompt не должна валить граф.
        serialized = repr(value)
    return f"[{title.upper()}]\n{truncate_text(serialized, max_chars)}\n"


def dedupe_lines(text: str, *, max_seen: int = 4_096) -> str:
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
