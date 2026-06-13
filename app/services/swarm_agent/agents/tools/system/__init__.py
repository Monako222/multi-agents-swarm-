"""Фабрика системных инструментов роя.

System-tools доступны каждому агенту: сохранить факты, обновить контекст,
зарегистрировать артефакт, записать ошибку, передать управление или завершить
ответ. Внешние инструменты (web, docs и т.д.) подключаются модульно.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import Field, create_model

from app.services.swarm_agent.agents.registry import FINAL_NODE
from app.services.swarm_agent.agents.tools.base import ArgsModel, make_async_tool
from app.services.swarm_agent.agents.tools.system.schemas import (
    ArtifactArgs,
    ContextArgs,
    ErrorArgs,
    FindingsArgs,
    FinishArgs,
)
from app.services.swarm_agent.graph.state import ErrorRecord, File, PendingTransfer
from app.services.swarm_agent.types import (
    ErrorSeverity,
    FileKind,
    JsonObject,
    OutputMode,
    ToolCategory,
    TransferProtocol,
)
from app.services.swarm_agent.utils import clean_payload


def _back(caller: str, update: dict[str, Any]) -> Command:
    """Служебный возврат управления агенту без смены узла графа."""
    
    return Command(update=update, goto=caller)


def _prefix(caller: str) -> str:
    """Сверхбыстрая генерация PascalCase-префикса для Pydantic."""
    
    # Используем нативный capitalize, это быстрее и чище срезов
    return "".join(p.capitalize() for p in caller.split("_") if p)


def build_system_tools(
    caller: str, 
    peers: tuple[str, ...] = ()
) -> list[BaseTool]:
    """Фабрика базовых инструментов для текущего агента."""

    async def save_findings(
        goal: str | None = None,
        notes: list[str] | None = None,
        data: JsonObject | None = None,
    ) -> Command:
        """Сохранение фактов и заметок в общую память роя."""
        
        update: dict[str, Any] = {"metrics": {"tool_calls": 1}}
        
        # Элегантная фильтрация пустых значений
        if patch := clean_payload(goal=goal, notes=notes):
            update["space"] = patch
            
        if data is not None:
            update["data"] = {"json_data": data}
            
        return _back(caller, update)

    async def update_context(
        query: str | None = None,
        intent: str | None = None,
        subject: str | None = None,
        tags: list[str] | None = None,
        lang: str | None = None,
    ) -> Command:
        """Обновление нормализованного семантического контекста."""
        
        patch = clean_payload(
            query=query,
            intent=intent,
            subject=subject,
            tags=tags,
            lang=lang.lower() if lang else None,
        )
        
        return _back(caller, {
            "context": patch, 
            "metrics": {"tool_calls": 1}
        })

    async def submit_artifact(
        file_uri: str,
        description: str,
        kind: FileKind = FileKind.DOC,
        content_preview: str | None = None,
    ) -> Command:
        """Регистрация итогового выходного артефакта (файла)."""
        
        artifact = File(
            id=file_uri,
            uri=file_uri,
            desc=description,
            kind=kind,
            content=content_preview or "",
        )
        
        return _back(caller, {
            "out_files": [artifact], 
            "metrics": {"tool_calls": 1}
        })

    async def report_error(
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ) -> Command:
        """Логирование recoverable-ошибки в журнал графа."""
        
        error = ErrorRecord(
            source=caller, 
            message=message, 
            severity=severity
        )
        
        return _back(caller, {
            "errors": [error], 
            "metrics": {"tool_calls": 1}
        })

    async def finish(final: str) -> Command:
        """Полная остановка графа и отправка ответа пользователю."""
        
        return Command(
            update={
                "workspace": {
                    "draft_answer": final, 
                    "final_answer": final
                },
                "active_node": FINAL_NODE,
                "pending_transfer": None,
                "loops": 0,
                "is_final": True,
                "metrics": {"tool_calls": 1},
            },
            goto=FINAL_NODE,
        )

    # -----------------------------------------------------------------------
    # Базовый каталог, доступный абсолютно любому агенту
    # -----------------------------------------------------------------------
    tools: list[BaseTool] = [
        make_async_tool(
            coroutine=save_findings,
            name="save_findings",
            description=(
                "Save compact reusable facts, notes, or structured JSON. "
                "Safe to call in parallel."
            ),
            args_schema=FindingsArgs,
            category=ToolCategory.SYSTEM,
            parallel_safe=True,
            routes=False,
        ),
        make_async_tool(
            coroutine=update_context,
            name="update_context",
            description=(
                "Update stable request metadata: query, intent, tags, etc. "
                "Safe to call in parallel."
            ),
            args_schema=ContextArgs,
            category=ToolCategory.SYSTEM,
            parallel_safe=True,
            routes=False,
        ),
        make_async_tool(
            coroutine=submit_artifact,
            name="submit_artifact",
            description="Register a generated output artifact. Do not invent URIs.",
            args_schema=ArtifactArgs,
            category=ToolCategory.SYSTEM,
            parallel_safe=True,
            routes=False,
        ),
        make_async_tool(
            coroutine=report_error,
            name="report_error",
            description="Record a degraded execution note, then continue or finish.",
            args_schema=ErrorArgs,
            category=ToolCategory.SYSTEM,
            parallel_safe=True,
            routes=False,
        ),
        make_async_tool(
            coroutine=finish,
            name="finish",
            description=(
                "Finish the task and provide final answer. "
                "Routing tool: call at most once per turn."
            ),
            args_schema=FinishArgs,
            category=ToolCategory.SYSTEM,
            parallel_safe=False,
            routes=True,
        ),
    ]

    # -----------------------------------------------------------------------
    # Инструмент передачи эстафеты (Handoff / Transfer)
    # Генерируется динамически только если есть разрешенные соседи (peers)
    # -----------------------------------------------------------------------
    uniq_peers = tuple(sorted(set(peers)))
    if not uniq_peers:
        return tools

    prefix = _prefix(caller)
    peer_enum = Enum(
        f"{prefix}TransferTarget",
        {p.upper(): p for p in uniq_peers},
        type=str,
    )
    
    transfer_args = create_model(
        f"{prefix}TransferArgs",
        __base__=ArgsModel,
        target_agent=(
            peer_enum, 
            Field(..., description="Target peer.")
        ),
        task_description=(
            str, 
            Field(..., min_length=1, max_length=2_000)
        ),
        accepted_output_modes=(
            list[OutputMode] | None,
            Field(default=None, description="Output modes.")
        ),
    )

    async def transfer(
        target_agent: Any,
        task_description: str,
        accepted_output_modes: list[OutputMode] | None = None,
    ) -> Command:
        """Передача управления (handoff) другому агенту из списка peers."""
        
        target = str(getattr(target_agent, "value", target_agent))
        pending = PendingTransfer(
            target_agent=target,
            requested_by=caller,
            protocol=TransferProtocol.LOCAL,
            task_description=task_description,
            accepted_output_modes=tuple(
                accepted_output_modes or [
                    OutputMode.TEXT, 
                    OutputMode.FILE
                ]
            ),
        )
        
        return Command(
            update={
                "active_node": target,
                "pending_transfer": pending,
                "history": [f"{caller} -> {target}: {task_description}"],
                "loops": 0,
                "metrics": {"tool_calls": 1},
            },
            goto=target,
        )

    tools.append(
        make_async_tool(
            coroutine=transfer,
            name="transfer",
            description=(
                "Transfer control to an allowed peer with a compact task description. "
                "Routing tool: call at most once per turn."
            ),
            args_schema=transfer_args,
            category=ToolCategory.SYSTEM,
            parallel_safe=False,
            routes=True,
        )
    )
    
    return tools
