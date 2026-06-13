"""Финализация и восстановление ответа графа.

Срабатывает на терминальных узлах графа для защиты от сбоев:
останавливает зацикливания, восстанавливает забытые вызовы finish() 
и нормализует итоговый workspace перед отдачей результата пользователю.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.services.swarm_agent.agents import FINAL_NODE
from app.services.swarm_agent.graph.budget import render_message_for_memory
from app.services.swarm_agent.graph.state import ErrorRecord, SwarmState
from app.services.swarm_agent.types import ErrorSeverity


def _field(obj: Any, key: str, default: Any = "") -> Any:
    """Универсальное итерирование: достает поле из Pydantic-модели или dict."""
    
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _content_text(msg: Any) -> str:
    """Безопасное извлечение текста из сообщения для recovery/finalizer."""
    
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
        
    return render_message_for_memory(msg, max_chars=20_000)


async def recover_unstructured_answer_node(
    state: SwarmState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Спасти ответ, если LLM написала plain text вместо вызова finish()."""
    
    del config
    text = ""
    
    # Мгновенный доступ к последнему сообщению через моржовый оператор
    if msgs := state.get("messages"):
        text = _content_text(msgs[-1]).strip()
        
    text = text or "Задача завершилась без явного финального ответа."
    
    logger.bind(node="recovery").warning(
        "Recovered plain-text answer without finish tool"
    )
    
    return {
        "workspace": {
            "draft_answer": text, 
            "final_answer": text
        },
        "errors": [
            ErrorRecord(
                source="recovery",
                severity=ErrorSeverity.WARNING,
                message="Recovered plain-text answer without finish tool.",
            )
        ],
        "active_node": FINAL_NODE,
        "pending_transfer": None,
        "is_final": True,
    }


async def loop_guard_node(
    state: SwarmState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Безопасно завершить граф при достижении лимита циклов."""

    del config
    text = ""
    workspace = state.get("workspace")
    if workspace:
        text = (
            str(_field(workspace, "final_answer", "") or _field(workspace, "draft_answer", ""))
            .strip()
        )
    if not text and (msgs := state.get("messages")):
        text = _content_text(msgs[-1]).strip()
    text = text or "Достигнут лимит шагов. Частичный ответ не сформирован."

    return {
        "workspace": {"draft_answer": text, "final_answer": text},
        "errors": [
            ErrorRecord(
                source="loop_guard",
                severity=ErrorSeverity.WARNING,
                message="Loop or total step limit reached.",
            )
        ],
        "active_node": FINAL_NODE,
        "pending_transfer": None,
        "is_final": True,
    }


async def finalizer_node(
    state: SwarmState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Нормализовать финальный workspace перед END."""

    del config
    workspace = state.get("workspace")
    final_answer = ""
    draft_answer = ""
    if workspace:
        final_answer = str(_field(workspace, "final_answer", "") or "").strip()
        draft_answer = str(_field(workspace, "draft_answer", "") or "").strip()

    text = final_answer or draft_answer
    if not text and (msgs := state.get("messages")):
        text = _content_text(msgs[-1]).strip()
    text = text or "Задача завершена без финального ответа."

    return {
        "workspace": {"draft_answer": draft_answer or text, "final_answer": text},
        "active_node": FINAL_NODE,
        "pending_transfer": None,
        "is_final": True,
    }
