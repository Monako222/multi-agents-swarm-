"""Маршрутизация после LLM-узла.

Быстрый и детерминированный выбор следующего шага графа 
(tools, recovery, loop_guard или final).
"""

from __future__ import annotations

from collections.abc import Callable

from app.services.swarm_agent.agents import FINAL_NODE, LOOP_GUARD_NODE, RECOVERY_NODE
from app.services.swarm_agent.graph.state import SwarmState


def _limit_reached(
    state: SwarmState,
    local_loop_limit: int | None,
) -> bool:
    """Проверяет бизнес-лимиты графа за O(1) до срабатывания recursion_limit."""
    
    if int(state.get("total_steps") or 0) >= 64:
        return True
        
    max_loops = local_loop_limit or 8
    return int(state.get("loops") or 0) >= max_loops


def make_agent_router(
    *,
    tools_node: str,
    local_loop_limit: int | None = None,
) -> Callable[[SwarmState], str]:
    """Фабрика роутера для агента.
    
    Гарантирует, что готовый finish или plain-text ответ 
    будут сохранены до ухода в аварийный loop-guard.
    """

    def _route(state: SwarmState) -> str:
        
        # 1. Мгновенный выход, если задача уже завершена
        if state.get("is_final"):
            return FINAL_NODE

        # 2. Оптимизированное извлечение последнего сообщения ровно 1 раз
        msg = None
        if msgs := state.get("messages"):
            msg = msgs[-1]

        if msg is not None:
            
            # Любой tool_call обязан получить свой ToolMessage. 
            # Иначе протокол провайдера будет нарушен на следующем шаге.
            if getattr(msg, "tool_calls", None):
                return tools_node

            # Спасаем plain-text ответ, если LLM забыла вызвать finish()
            if str(getattr(msg, "content", "")).strip():
                return RECOVERY_NODE

        # 3. Защита от бесконечного зацикливания агента
        if _limit_reached(state, local_loop_limit):
            return LOOP_GUARD_NODE
            
        # 4. Fallback-маршрутизация
        return RECOVERY_NODE

    return _route


# Alias оставлен для обратной совместимости внешних тестов и импортов
route_agent_output = make_agent_router
