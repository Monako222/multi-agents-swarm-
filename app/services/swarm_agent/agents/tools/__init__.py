"""Каталог инструментов роя.

Пакет разделён по категориям: ``system`` для handoff/state/finalize, а также
заготовки ``web``, ``docs``, ``vision``, ``audio`` и ``video`` для будущих
интеграций. Экспорт ленивый, чтобы import лёгких модулей не тянул LangChain.
"""

from __future__ import annotations

__all__ = ["build_system_tools", "clear_tools_cache", "get_agent_tools"]


def __getattr__(name: str):
    """Ленивая загрузка factory/cache, зависящих от LangChain."""
    if name in {"clear_tools_cache", "get_agent_tools"}:
        from app.services.swarm_agent.agents.tools.cache import clear_tools_cache, get_agent_tools

        return {
            "clear_tools_cache": clear_tools_cache,
            "get_agent_tools": get_agent_tools,
        }[name]
    if name == "build_system_tools":
        from app.services.swarm_agent.agents.tools.system import build_system_tools

        return build_system_tools
    raise AttributeError(name)
