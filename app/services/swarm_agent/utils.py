"""Мелкие runtime-утилиты без привязки к LangGraph.
Сверхбыстрые операции на критическом пути: чтение сообщений, 
нормализация списков, обрезка памяти (state budget) и 
глубокое слияние параллельных tool-update.
"""


import orjson

from typing import Any, Final
from pydantic import BaseModel
from collections.abc import Mapping, Sequence




_LIST_KEYS: Final[
    frozenset[str]
] = frozenset({
    "out_files",
    "messages", 
    "errors", 
    "history", 
    "in_files",
})



_PATCH_KEYS: Final[
    frozenset[str]
] = frozenset({
    "workspace",
    "context", 
    "space", 
    "data",
})



_ATOMIC_TYPES: Final[
    tuple[type, ...]
] = (
    bytes, 
    BaseModel,
    bytearray, 
    Mapping,

)






def clean_payload(**kwargs: Any) -> dict[str, Any]:
    """Быстрый фильтр None-значений для 
    защиты стейта от затирания."""
    
    return {
        k: v for k, v 
        in kwargs.items() 
        if v is not None
    }






def as_list(value: Any) -> list[Any]:
    """Приводит скаляры к списку за O(1).
    Использует кэширование типа для исключения 
    двойных вызовов C-API."""
    
    if value is None:
        return []
        
    # Единожды дергаем 
    # для получения типа
    v_type = type(value)
    if v_type is list:
        return value
        
    if v_type is tuple:
        return list(value)
        
    # Базовые и составные типы 
    # оборачиваем целиком, защищая от распада
    if (
        v_type is str or v_type is dict or 
        isinstance(value, _ATOMIC_TYPES)
    ):
        return [value]
        
    if isinstance(value, Sequence):
        return list(value)
        
    return [value]






def message_role(msg: Any) -> str:
    """Нормализация роли из
    LangChain/OpenAI-сообщений."""
    
    if type(msg) is dict:
        return str(
            msg.get("type") or 
            msg.get("role") or "dict"
        ).lower()
        
        
    return str(
        getattr(msg, "type", None) 
        or getattr(msg, "role", None) 
        or msg.__class__.__name__
    ).lower()






def message_content(msg: Any) -> str:
    """Безопасное извлечение текста 
    с обходом мультимодальных блоков."""
    
    content = getattr(msg, "content", "")
    return (
        content 
        if type(content) is str 
        else str(content)
    )






def clip(
    text: str, 
    limit: int, 
    suffix: str = "..."
) -> str:
    """Прямая обрезка строки 
    с жестким лимитом для логов 
    и state)."""
    
    if limit <= 0 or len(text) <= limit:
        return text
    
    return text[: max(0, limit - len(suffix))].rstrip() + suffix






def clip_tail(
    text: str, 
    limit: int, 
    prefix: str = "...\n"
) -> str:
    """Обратная обрезка: сохраняет 
    самый свежий хвост длинной
    памяти."""
    
    if limit <= 0 or len(text) <= limit:
        return text
        
    return prefix + text[-max(0, limit - len(prefix)):].lstrip()






def to_json(value: Any) -> str:
    """Молниеносная сериализация 
    объекта для prompt-контекста."""
    
    if isinstance(value, BaseModel):
        return value.model_dump_json(
            exclude_unset=True,
            exclude_none=True
        )
        
    return orjson.dumps(
        value, default=str, 
        option=orjson.OPT_SORT_KEYS
    ).decode("utf-8")






def state_part(
    title: str, 
    value: Any, 
    max_chars: int
) -> str:
    """Форматирует часть состояния.
    Пропускает пустые значения и 
    обрабатывает объекты."""
    
    if not value:
        return ""

    try: 
        text = to_json(value)
    except Exception: 
        text = repr(value)

    body = clip(text, max_chars)
    return f"[{title.upper()}]\n{body}\n"








def uniq_lines(text: str, max_seen: int = 4_096) -> str:
    """Очищает текст от дублей. Использует 
    кэширование всех указателей для
    производительности."""
    
    out: list[str] = []
    seen: set[str] = set()

    seen_clear  = seen.clear
    out_append = out.append
    seen_add = seen.add
    
    for line in text.splitlines():
        key = line.strip()
        
        if not key:
            out_append(line)
            continue
            
        if key in seen:
            continue
            
        if len(seen) >= max_seen:
            seen_clear()
            
        seen_add(key)
        out_append(line)
        
    return "\n".join(out)









def safe_error_text(
    exc: BaseException, 
    limit: int = 1_500
) -> str:
    """Сжимает трейсбэк 
    ошибки для логов и 
    снапшотов."""
    
    return clip(
        str(exc).strip() or 
        exc.__class__.__name__, 
        limit
    )









def _merge_patch(
    left: Mapping[str, Any], 
    right: Mapping[str, Any]
) -> dict[str, Any]:
    """Рекурсивное глубокое слияние 
    словарей без потери списков."""
    
    merged = dict(left)
    for k, v in right.items():
        if k not in merged:
            merged[k] = v
            continue
            
        old = merged[k]
        
        is_v_dict = (
            type(v) is dict or 
            isinstance(v, Mapping)
        )
        
        is_old_dict = (
            type(old) is dict or 
            isinstance(old, Mapping)
        )
        
        if is_old_dict and is_v_dict:
            merged[k] = _merge_patch(old, v)
            
        elif (
            type(old) is list 
            or type(v) is list 
            or isinstance(old, list) 
            or isinstance(v, list)
        ):
            merged[k] = as_list(old) + as_list(v)
            
        else: merged[k] = v
            
    return merged







def _merge_metrics(
    left: Mapping[str, Any], 
    right: Mapping[str, Any]
) -> dict[str, Any]:
    """Суммирует числовые
    метрики за O(K)."""
    
    merged = dict(left)
    for k, v in right.items():
        if type(v) is int or type(v) is float:
            try: 
                merged[k] = int(merged.get(k, 0)) + int(v)
            except (ValueError, TypeError):
                merged[k] = v
        else:
            merged[k] = v
            
    return merged








def merge_update(
    dst: dict[str, Any], 
    src: Mapping[str, Any]
) -> dict[str, Any]:
    """Точка сборки: объединяет параллельные 
    tool_updates в единый патч. Использует кэширование 
    методов и цепочки `elif` для обхода лишних сравнений 
    на уровне байткода CPython."""
    
    dst_get = dst.get 
    for k, v in src.items():
        if v is None:
            continue
            
        if k in _LIST_KEYS:
            dst[k] = as_list(dst_get(k)) + as_list(v)
            
        elif k == "metrics":
            if type(v) is dict or isinstance(v, Mapping):
                base = dst_get(k)
                base_dict = (
                    base 
                    if (
                        type(base) is dict or 
                        isinstance(base, Mapping)
                    ) 
                    else {}
                )
                dst[k] = _merge_metrics(base_dict, v)
                
        elif k in _PATCH_KEYS:
            if type(v) is dict or isinstance(v, Mapping):
                base = dst_get(k)
                base_dict = (
                    base 
                    if (
                        type(base) is dict or 
                        isinstance(base, Mapping)
                    ) 
                    else {}
                )
                dst[k] = _merge_patch(base_dict, v)
                
        else:
            dst[k] = v
            
    return dst



