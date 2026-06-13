"""Инициализация run внутри LangGraph.

Подготавливает начальное состояние роя: извлекает стартовый запрос 
пользователя и очищает временные поканальные данные, чтобы разные 
запуски в рамках одного потока (thread) проходили без смешивания артефактов.
"""

from __future__ import annotations

from typing import Any

from app.services.swarm_agent.graph.state import Context, SwarmState, replace_list, replace_value


def _message_content(msg: Any) -> str:
    """Безопасное извлечение текста из пользовательского сообщения."""
    
    content = getattr(msg, "content", "")
    return content if isinstance(content, str) else str(content)


def extract_initial_query(state: SwarmState) -> str:
    """Извлекает самый свежий запрос пользователя из сообщений или контекста."""
    
    # 1. Приоритетно ищем последний промпт от человека (обратный проход)
    if msgs := state.get("messages"):
        for msg in reversed(msgs):
            role = str(
                getattr(msg, "type", None) or getattr(msg, "role", "")
            ).lower()
            
            if role in {"human", "user"}:
                if text := _message_content(msg).strip():
                    return text

    # 2. Фоллбэк: берем из контекста (поддерживает словари и объекты Context)
    if ctx := state.get("context"):
        if isinstance(ctx, dict) and (q := ctx.get("query")):
            return str(q).strip()
            
        if isinstance(ctx, Context) and ctx.query:
            return str(ctx.query).strip()

    return ""


def init_swarm(
    state: SwarmState, 
    *, 
    entry_node: str
) -> dict[str, Any]:
    """Инициализация нового цикла (run) без потери долговременной памяти.

    Сбрасывает все per-run каналы (файлы, ошибки, историю переходов),
    предотвращая отравление нового контекста результатами прошлого запроса
    в рамках одного и того же persistent thread.
    """
    
    update: dict[str, Any] = {
        "active_node": entry_node,
        "loops": 0,
        "total_steps": 0,
        "pending_transfer": None,
        
        "workspace": {
            "draft_answer": "", 
            "final_answer": ""
        },
        
        # Полная зачистка списков через маркеры замены LangGraph-редьюсеров
        "errors": replace_list([]),
        "history": replace_list([]),
        "out_files": replace_list([]),
        
        "data": {
            "peers": replace_list([]),
            "files": replace_list([]),
            "json_data": replace_value({}),
        },
        
        "space": {
            "goal": "", 
            "step": "", 
            "brief": "", 
            "notes": replace_list([])
        },
        
        "is_final": False,
    }

    # Мгновенный патч контекста при обнаружении нового запроса
    if query := extract_initial_query(state):
        update["context"] = {
            "query": query,
            "intent": "",
            "subject": "",
            "lang": "",
            "tags": replace_list([]),
        }
        
    return update
