"""Кэш tool-схем по агенту и категориям. 
Схемы передаются в каждый LLM-вызов. Переиспользование 
объектов снижает churn и помогает prompt caching. 
Ключ учитывает caller, peers и категории.
"""



from functools import lru_cache

from langchain_core.tools import BaseTool
from app.services.swarm_agent.exceptions import RegistryValidationError
from app.services.swarm_agent.types import ToolCategory

from .system import build_system_tools
from .web import build_web_tools


_FACTORIES = {
    # ToolCategory.AUDIO: build_audio_tools,
    # ToolCategory.DOCS: build_docs_tools,
    # ToolCategory.VIDEO: build_video_tools,
    # ToolCategory.VISION: build_vision_tools,
    ToolCategory.WEB: build_web_tools,
}


_KNOWN_CATS = frozenset(
    cat.value for cat 
    in ToolCategory
)




def _assert_unique(
    tools: list[BaseTool], 
    caller: str
) -> None:
    """Мгновенная проверка 
    дубликатов."""
    
    seen: set[str] = set()
    dupes: set[str] = set()
    
    for t in tools:
        if t.name in seen:
            dupes.add(t.name)
        else:
            seen.add(t.name)
            
    if dupes:
        joined = ", ".join(sorted(dupes))
        raise RegistryValidationError(
            f"Agent {caller!r} tools "
            f"duplicate : {joined}"
        )






@lru_cache(maxsize=512)
def get_agent_tools(
    caller: str,
    peers: tuple[str, ...] = (),
    cats: tuple[str, ...] = (),
) -> tuple[BaseTool, ...]:
    """Кэшированный набор инструментов 
    агента.Системные добавляются всегда,
    остальные — по запросу.
    """
    
    uniq_peers = tuple(sorted(set(peers)))
    uniq_cats = tuple(sorted(set(cats)))
    
    if unknown := set(uniq_cats) - _KNOWN_CATS:
        joined = ", ".join(sorted(unknown))
        raise RegistryValidationError(
            f"Agent {caller!r} unknown "
            f"categories: {joined}"
        )

    # Всегда стартуем 
    # с системных инструментов 
    tools: list[BaseTool] = list(
        build_system_tools(
            caller, uniq_peers
        )
    )
    
    for cat in uniq_cats:
        if cat == ToolCategory.SYSTEM.value:
            continue
            
        if factory := _FACTORIES.get(cat):
            tools.extend(factory(caller))

    _assert_unique(tools, caller)
    return tuple(tools)


def clear_tools_cache() -> None:
    get_agent_tools.cache_clear()
