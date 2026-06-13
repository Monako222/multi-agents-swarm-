"""Сборка LangGraph-графа роя.

Динамически компилирует маршруты и узлы на основе реестра агентов.
Узлы инструментов (tool nodes) маршрутизируются динамически через Command(goto=...),
что исключает конфликты со статической маршрутизацией графа.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.services.swarm_agent.agents import (
    FINAL_NODE,
    LOOP_GUARD_NODE,
    RECOVERY_NODE,
    AgentRegistry,
    agent_registry,
)
from app.services.swarm_agent.agents.tools import get_agent_tools
from app.services.swarm_agent.config import Settings, get_settings
from app.services.swarm_agent.graph.bootstrap import init_swarm
from app.services.swarm_agent.graph.runtime import (
    AgentNode,
    ToolExecutorNode,
    finalizer_node,
    loop_guard_node,
    make_agent_router,
    recover_unstructured_answer_node,
)
from app.services.swarm_agent.graph.state import SwarmState
from app.services.swarm_agent.llm import LLMHub, LazyLLMHub


def build_swarm_graph(
    *,
    checkpointer: Any = None,
    settings: Settings | None = None,
    registry: AgentRegistry = agent_registry,
    hub: LLMHub | LazyLLMHub | None = None,
) -> Any:
    """Компилирует StateGraph роя со всеми агентами и fallback-узлами."""
    
    cfg = settings or get_settings()
    llm_hub = hub or LazyLLMHub(settings=cfg)
    builder = StateGraph(SwarmState)

    # 1. Узел инициализации: очищает временную память перед каждым run
    async def _init(state: SwarmState) -> dict[str, Any]:
        return init_swarm(state, entry_node=registry.entry_node)

    builder.add_node("init", _init)
    builder.add_edge(START, "init")
    builder.add_edge("init", registry.entry_node)

    # 2. Динамическая генерация узлов агентов и их инструментов
    for spec in registry.specs():
        
        # Кэшируем имя за O(1), чтобы не дергать атрибут Pydantic-модели
        name = spec.name
        tools_node = f"{name}_tools"
        
        peers = registry.peer_names(name)
        tools = get_agent_tools(name, peers, spec.tools)
        valid_gotos = (name, FINAL_NODE, *peers)

        # Главный LLM-узел агента
        builder.add_node(
            name,
            AgentNode(
                spec, 
                registry=registry, 
                settings=cfg, 
                hub=llm_hub
            ),
        )
        
        # Изолированный исполнитель инструментов агента
        builder.add_node(
            tools_node,
            ToolExecutorNode(
                caller_name=name,
                tools=tools,
                valid_gotos=valid_gotos,
                settings=cfg,
                local_loop_limit=spec.max_local_loops,
            ),
        )
        
        # Маршрутизатор: направляет флоу после ответа LLM
        builder.add_conditional_edges(
            name,
            make_agent_router(
                tools_node=tools_node,
                settings=cfg,
                local_loop_limit=spec.max_local_loops,
            ),
            {
                tools_node: tools_node,
                RECOVERY_NODE: RECOVERY_NODE,
                LOOP_GUARD_NODE: LOOP_GUARD_NODE,
                FINAL_NODE: FINAL_NODE,
            },
        )

    # 3. Терминальные (fallback) узлы для безопасного завершения графа
    builder.add_node(RECOVERY_NODE, recover_unstructured_answer_node)
    builder.add_edge(RECOVERY_NODE, FINAL_NODE)

    builder.add_node(LOOP_GUARD_NODE, loop_guard_node)
    builder.add_edge(LOOP_GUARD_NODE, FINAL_NODE)

    builder.add_node(FINAL_NODE, finalizer_node)
    builder.add_edge(FINAL_NODE, END)

    # Финальная компиляция в исполняемый граф
    return builder.compile(
        checkpointer=checkpointer, 
        debug=cfg.debug
    )
