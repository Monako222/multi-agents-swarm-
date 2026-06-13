"""Базовые примитивы для построения инструментов.
Каждый tool помечается metadata: можно ли его выполнять параллельно 
и меняет ли он маршрутизацию графа. ToolExecutor читает эти флаги, 
чтобы не смешивать конфликтующие transfer/finish в одном turn.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict
from app.services.swarm_agent.types import ToolCategory



class ToolMeta(TypedDict):
    """Metadata, читаемая executor-ом
    без привязки к категории."""
    
    routes: bool
    category: str
    parallel_safe: bool
    requires_tool_call_id: NotRequired[bool]




class ArgsModel(BaseModel):
    """Базовая строгая схема 
    аргументов tool-call. """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        validate_by_alias=True,
        validate_default=True,
        validate_by_name=True,
        extra="ignore",
    )





def make_async_tool(
    *,
    name: str,
    coroutine: Any,
    description: str,
    args_schema: type[BaseModel],
    category: ToolCategory,
    parallel_safe: bool,
    routes: bool,
    requires_tool_call_id: bool = False,
) -> BaseTool:
    """Создать async StructuredTool
    без sync-func fallback."""
    
    return StructuredTool.from_function(
        name=name,
        coroutine=coroutine,
        description=description,
        args_schema=args_schema,
        metadata={
            "category": category.value,
            "parallel_safe": parallel_safe,
            "routes": routes,
            "requires_tool_call_id": requires_tool_call_id,
        },
    )



def is_parallel_safe(tool: BaseTool) -> bool:
    """Можно ли выполнять инструмент 
    параллельно с non-routing tools."""

    return bool(
        tool.metadata and 
        tool.metadata.get(
            "parallel_safe"
        )
    )




def is_routing_tool(tool: BaseTool) -> bool:
    """Меняет ли инструмент маршрутизацию 
    графа через Command.goto."""
    
    return bool(
        tool.metadata and 
        tool.metadata.get(
            "routes"
        )
    )


def requires_tool_call_id(tool: BaseTool) -> bool:
    """Нужно ли передавать tool-call envelope для InjectedToolCallId."""

    return bool(
        tool.metadata
        and tool.metadata.get(
            "requires_tool_call_id"
        )
    )
