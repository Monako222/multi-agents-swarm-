"""LangGraph builder и bootstrap.

Обычный import этого пакета не компилирует граф. ``create_graph`` нужен для
LangGraph Platform и создаёт singleton-граф лениво только при запуске платформы.
"""

from __future__ import annotations

from functools import lru_cache

__all__ = ["build_swarm_graph", "create_graph", "extract_initial_query", "graph", "init_swarm"]


@lru_cache(maxsize=1)
def _platform_graph():
    """Собрать process-wide граф для LangGraph Platform."""
    from app.services.swarm_agent.graph.builder import build_swarm_graph

    return build_swarm_graph()


def create_graph():
    """Factory-entrypoint для ``langgraph.json``."""
    return _platform_graph()


def __getattr__(name: str):
    """Совместимость для окружений, которые всё ещё ожидают атрибут graph."""
    if name == "graph":
        return _platform_graph()
    if name == "build_swarm_graph":
        from app.services.swarm_agent.graph.builder import build_swarm_graph

        return build_swarm_graph
    if name in {"extract_initial_query", "init_swarm"}:
        from app.services.swarm_agent.graph.bootstrap import extract_initial_query, init_swarm

        return {
            "extract_initial_query": extract_initial_query,
            "init_swarm": init_swarm,
        }[name]
    raise AttributeError(name)
