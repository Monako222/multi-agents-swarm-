"""Сборка runtime-контекста для агента."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.services.swarm_agent.graph.state import SwarmState
from app.services.swarm_agent.utils import clip, state_part


def _space_prompt_view(value: Any) -> Any:
    """Изолирует episodic_memory из SPACE (она уходит отдельным сообщением)."""
    
    # Сверхбыстрая проверка на пустые структуры данных
    if not value:
        return value
        
    if isinstance(value, BaseModel):
        data = value.model_dump(
            exclude_unset=True, 
            exclude_none=True
        )
    elif isinstance(value, dict):
        data = dict(value)
    else:
        return value
        
    data.pop("episodic_memory", None)
    return data


def format_runtime_context(
    state: SwarmState,
    *,
    part_char_limit: int,
    data_char_limit: int,
    file_char_limit: int,
    total_char_limit: int,
) -> str:
    """Собрать общий runtime-контекст, видимый каждому агенту.

    В prompt попадают только bounded-срезы каналов. Полный state остаётся в
    checkpointer, а LLM получает компактную, актуальную картину выполнения.
    """
    
    # Безопасное извлечение с фоллбэками
    errors = state.get("errors") or []
    history = state.get("history") or []

    # Декларативная сборка частей контекста
    parts = [
        state_part(
            "context", 
            state.get("context"), 
            max_chars=part_char_limit
        ),
        state_part(
            "space", 
            _space_prompt_view(state.get("space")), 
            max_chars=part_char_limit
        ),
        state_part(
            "data", 
            state.get("data"), 
            max_chars=data_char_limit
        ),
        state_part(
            "in_files", 
            state.get("in_files"), 
            max_chars=file_char_limit
        ),
        state_part(
            "out_files", 
            state.get("out_files"), 
            max_chars=file_char_limit
        ),
        state_part(
            "workspace", 
            state.get("workspace"), 
            max_chars=part_char_limit
        ),
        state_part(
            "pending_transfer",
            state.get("pending_transfer"),
            max_chars=part_char_limit,
        ),
        state_part(
            "errors", 
            errors[-20:], 
            max_chars=part_char_limit
        ),
        state_part(
            "history", 
            history[-50:], 
            max_chars=part_char_limit
        ),
        state_part(
            "metrics", 
            state.get("metrics"), 
            max_chars=part_char_limit
        ),
    ]

    # Быстрая склейка готового списка за O(N)
    if text := "".join([p for p in parts if p]):
        return clip(text, total_char_limit)
        
    return "No runtime context yet."


def message_content_text(message: Any) -> str:
    """Достать текстовое содержимое message для fallback-финализации."""
    
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)
