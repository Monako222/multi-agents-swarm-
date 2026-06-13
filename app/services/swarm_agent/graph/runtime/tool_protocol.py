"""Низкоуровневый протокол tool-call execution.

Изолированная логика для нормализации вызовов, создания ToolMessage 
и жесткой защиты управляющего стейта графа от вмешательства 
обычных (non-routing) инструментов.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.services.swarm_agent.graph.state import ErrorRecord
from app.services.swarm_agent.utils import as_list, clip
from app.services.swarm_agent.types import ErrorSeverity

# Множество заморожено для O(1) проверок
_CONTROL_KEYS: Final[frozenset[str]] = frozenset({
    "active_node", 
    "pending_transfer", 
    "is_final", 
    "loops", 
    "total_steps"
})

FINISH_TOOL: Final[str] = "finish"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Нормализованный вызов инструмента из AIMessage.tool_calls."""

    index: int
    name: str
    args: Mapping[str, Any]
    call_id: str


@dataclass(slots=True)
class ToolOutcome:
    """Обертка результата одного tool-call до финального слияния."""

    index: int
    update: dict[str, Any]
    goto: str
    routes: bool = False


def extract_tool_call(call: Any, *, index: int) -> ToolCall:
    """Сверхбыстрая нормализация dict/object структуры к единому интерфейсу."""
    
    if isinstance(call, Mapping):
        name = str(call.get("name") or "")
        args = call.get("args") or {}
        cid = str(call.get("id") or f"call_{index}_{name or 'unknown'}")
    else:
        name = str(getattr(call, "name", ""))
        args = getattr(call, "args", {})
        cid = str(getattr(call, "id", f"call_{index}_{name or 'unknown'}"))

    return ToolCall(
        index=index,
        name=name,
        args=args if isinstance(args, Mapping) else {},
        call_id=cid,
    )


def command_update(cmd: Command) -> dict[str, Any]:
    """Безопасное извлечение словаря update из Command."""
    
    upd = getattr(cmd, "update", None) or {}
    
    if isinstance(upd, Mapping):
        return dict(upd)
        
    try:
        return dict(upd)
    except Exception:  # noqa: BLE001
        return {}


def command_goto(cmd: Command, default: str) -> str:
    """Извлечение узла перехода (goto) с фоллбэком на caller."""
    
    if isinstance(g := getattr(cmd, "goto", None), str) and g:
        return g
    return default


def tool_message(content: str, *, name: str, call_id: str) -> ToolMessage:
    """Генерация закрывающего ToolMessage с валидным ID для графа."""
    
    kwargs = {
        "content": clip(content, 4_000),
        "name": name or "unknown_tool",
        "tool_call_id": call_id,
        "id": f"tool:{call_id}",
    }
    
    try:
        return ToolMessage(**kwargs)
    except TypeError:  # Фоллбэк для старых версий LangChain без поддержки id
        kwargs.pop("id", None)
        return ToolMessage(**kwargs)


def has_tool_message(update: Mapping[str, Any], call_id: str) -> bool:
    """Молниеносная проверка наличия закрывающего ToolMessage."""
    
    msgs = as_list(update.get("messages"))
    return any(getattr(m, "tool_call_id", None) == call_id for m in msgs)


def protocol_warning(call: ToolCall, text: str) -> ToolOutcome:
    """Формирование warning-исхода для закрытия tool_call_id без исполнения."""
    
    msg = tool_message(text, name=call.name, call_id=call.call_id)
    err = ErrorRecord(
        source="tool_executor",
        severity=ErrorSeverity.WARNING,
        message=f"Ignored tool call {call.name or 'unknown'}: {text}",
    )
    
    return ToolOutcome(
        index=call.index, 
        update={"messages": [msg], "errors": [err]}, 
        goto=""
    )


def strip_non_routing_state_changes(
    update: dict[str, Any],
    *,
    caller_name: str,
    tool_name: str,
) -> bool:
    """Жесткая защита маршрутизации.
    Удаляет попытки обычных инструментов подменить управляющий стейт.
    """
    
    # O(1) пересечение множеств — моментально находим запрещенные ключи
    if bad_keys := _CONTROL_KEYS & update.keys():
        for k in bad_keys:
            update.pop(k)
            
    removed = bool(bad_keys)

    # Изолированная проверка попытки перезаписать итоговый ответ
    ws = update.get("workspace")
    if isinstance(ws, Mapping) and "final_answer" in ws:
        update.pop("workspace", None)
        removed = True

    # Логируем нарушение безопасности
    if removed:
        err = ErrorRecord(
            source=caller_name,
            severity=ErrorSeverity.ERROR,
            message=f"Non-routing tool {tool_name!r} tried to modify routing/final state.",
        )
        update["errors"] = [*as_list(update.get("errors")), err]
        
    return removed
