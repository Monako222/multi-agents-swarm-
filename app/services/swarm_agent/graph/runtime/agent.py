"""LLM-backed агентский узел."""

from __future__ import annotations

from threading import RLock
from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.services.swarm_agent.agents import (
    FINAL_NODE,
    SWARM_PROTOCOL,
    AgentRegistry,
    AgentSpec,
    build_agent_prompt,
)
from app.services.swarm_agent.agents.tools import get_agent_tools
from app.services.swarm_agent.exceptions import MissingApiKeyError
from app.services.swarm_agent.graph.budget import (
    compact_messages,
    format_runtime_context,
    render_message_for_memory,
)
from app.services.swarm_agent.graph.runtime.retry import ainvoke_with_retries
from app.services.swarm_agent.graph.state import ErrorRecord, SwarmState
from app.services.swarm_agent.llm import LLMHub, LazyLLMHub
from app.services.swarm_agent.utils import clip, message_content, message_role, safe_error_text
from app.services.swarm_agent.types import ErrorSeverity


def _latest_user_query(state: SwarmState) -> str:
    """Взять последний human/user message, иначе fallback на context.query."""
    
    # Мгновенная проверка и извлечение через моржовый оператор
    if messages := state.get("messages"):
        for msg in reversed(messages):
            if message_role(msg) in {"human", "user"}:
                if text := message_content(msg).strip():
                    return text

    context = state.get("context")
    if isinstance(context, dict):
        return str(context.get("query") or "").strip()
        
    return str(getattr(context, "query", "") or "").strip()


def _has_human_message(messages: list[Any]) -> bool:
    """Мгновенно проверить наличие пользовательского сообщения в окне."""
    
    return any(message_role(m) in {"human", "user"} for m in messages)


def _remove_messages(ids: list[str]) -> list[Any]:
    """Сгенерировать объекты удаления за O(N)."""
    
    return [RemoveMessage(id=mid) for mid in ids]


def _bounded_history(messages: list[Any], *, char_limit: int) -> list[Any]:
    """Обрезать гигантские текстовые сообщения перед отправкой в LLM."""
    
    bounded: list[Any] = []
    
    for msg in messages:
        content = getattr(msg, "content", None)
        
        if not isinstance(content, str) or len(content) <= char_limit:
            bounded.append(msg)
            continue
            
        clone = getattr(msg, "model_copy", None)
        short = clip(content, char_limit)
        
        bounded.append(
            clone(update={"content": short}) if callable(clone) else msg
        )
        
    return bounded


class AgentNode:
    """Один LLM-агент роя.
    
    Узел лениво биндует LLM и инструменты. Системные сообщения создаются 
    один раз на объект узла, а tools берутся из сверхбыстрого кэша.
    """

    __slots__ = (
        "_hub",
        "_llm",
        "_lock",
        "_model_alias",
        "_registry",
        "_sys_msgs",
        "_tools",
        "name",
    )

    def __init__(
        self,
        spec: AgentSpec,
        *,
        registry: AgentRegistry,
        hub: LLMHub | LazyLLMHub | None = None,
    ) -> None:
        
        self.name = spec.name
        self._registry = registry
        self._hub = hub or LazyLLMHub()
        self._model_alias = spec.model_alias or "fast"
        
        self._tools = get_agent_tools(
            spec.name, 
            registry.peer_names(spec.name), 
            spec.tools
        )
        
        self._lock = RLock()
        self._llm: Any | None = None

        agent_prompt = build_agent_prompt(
            name=spec.name,
            role=spec.role,
            tasks=spec.tasks,
            peers=registry.peers(spec.name),
            rules=spec.rules,
        )
        
        self._sys_msgs = [
            SystemMessage(content=SWARM_PROTOCOL),
            SystemMessage(content=agent_prompt),
        ]

    def _hub_instance(self) -> LLMHub:
        """Вернуть реальный LLMHub (разрешить lazy-загрузку)."""
        
        if isinstance(self._hub, LazyLLMHub):
            return self._hub.get()
        return self._hub

    def _bound_llm(self) -> Any:
        """Выдать cached LLM с привязанными схемами инструментов."""
        
        if self._llm is not None:
            return self._llm
            
        with self._lock:
            if self._llm is not None:
                return self._llm
                
            base = self._hub_instance().get(self._model_alias)
            bind_kwargs: dict[str, Any] = {"tool_choice": "auto", "parallel_tool_calls": True}
            
            try:
                self._llm = base.bind_tools(self._tools, **bind_kwargs)
            except TypeError:  # fallback для провайдеров без поддержки ptc
                self._llm = base.bind_tools(self._tools, tool_choice="auto")
                
            return self._llm

    async def __call__(
        self, 
        state: SwarmState, 
        config: RunnableConfig
    ) -> dict[str, Any]:
        """Выполнить один reasoning/tool-calling шаг агента."""
        
        loops = int(state.get("loops") or 0) + 1
        total_steps = int(state.get("total_steps") or 0) + 1
        
        prev_mem = ""
        if space := state.get("space"):
            prev_mem = getattr(space, "episodic_memory", "")
            
        raw_history = list(state.get("messages") or [])

        compaction = compact_messages(
            raw_history,
            keep_last=12,
            previous_summary=prev_mem,
            max_summary_chars=8_000,
        )
        
        remove_msgs = _remove_messages(compaction.remove_ids)
        query = _latest_user_query(state)

        runtime_context = format_runtime_context(
            state,
            part_char_limit=4_000,
            data_char_limit=8_000,
            file_char_limit=6_000,
            total_char_limit=24_000,
        )
        
        # Декларативно собираем финальный массив сообщений
        llm_msgs = [*self._sys_msgs]
        
        if compaction.episodic_memory:
            llm_msgs.append(
                SystemMessage(
                    content=f"--- EPISODIC MEMORY ---\n{compaction.episodic_memory}"
                )
            )
            
        llm_msgs.append(
            SystemMessage(
                content=f"--- RUNTIME STATE ---\n{runtime_context}"
            )
        )
        
        if query and not _has_human_message(compaction.kept):
            llm_msgs.append(
                HumanMessage(
                    content=f"Текущий запрос пользователя:\n{query}"
                )
            )
            
        llm_msgs.extend(
            _bounded_history(
                compaction.kept, 
                char_limit=40_000,
            )
        )

        prompt_chars = sum(
            len(render_message_for_memory(m, max_chars=8_000)) 
            for m in llm_msgs
        )

        try:
            response, retries = await ainvoke_with_retries(
                self._bound_llm(),
                llm_msgs,
                config,
                node_name=self.name,
            )
            
        except MissingApiKeyError as exc:
            logger.bind(node=self.name).error("API key missing: {}", exc)
            return _fatal_node_update(
                self.name,
                "Сейчас не получилось обработать запрос. Попробуйте, пожалуйста, чуть позже.",
                exc,
                loops=loops,
                total_steps=total_steps,
            )
            
        except Exception as exc:  # noqa: BLE001
            logger.bind(node=self.name).exception("LLM node failed")
            return _fatal_node_update(
                self.name,
                "Сейчас не получилось обработать запрос. Попробуйте, пожалуйста, чуть позже.",
                exc,
                loops=loops,
                total_steps=total_steps,
            )

        # Формируем чистый стейт-апдейт за O(1)
        update: dict[str, Any] = {
            "messages": [*remove_msgs, response],
            "active_node": self.name,
            "loops": loops,
            "total_steps": total_steps,
            "metrics": {
                "llm_calls": 1,
                "retries": retries,
                "prompt_chars": prompt_chars,
                "completion_chars": len(
                    render_message_for_memory(response, max_chars=40_000)
                ),
                "compacted_messages": compaction.evicted_count,
            },
        }
        
        if compaction.episodic_memory != prev_mem:
            update["space"] = {"episodic_memory": compaction.episodic_memory}
            
        if compaction.unremovable_count:
            warn_msg = (
                f"Compacted {compaction.unremovable_count} messages without ids; "
                "they cannot be removed from checkpoint storage."
            )
            update["errors"] = [
                ErrorRecord(
                    source=self.name,
                    severity=ErrorSeverity.WARNING,
                    message=warn_msg,
                )
            ]
            
        return update


def _fatal_node_update(
    source: str,
    user_text: str,
    exc: BaseException,
    *,
    loops: int,
    total_steps: int,
) -> dict[str, Any]:
    """Сформировать recoverable terminal update при критическом сбое узла."""
    
    return {
        "workspace": {"draft_answer": user_text, "final_answer": user_text},
        "errors": [
            ErrorRecord(
                source=source,
                severity=ErrorSeverity.CRITICAL,
                message=safe_error_text(exc),
            )
        ],
        "loops": loops,
        "total_steps": total_steps,
        "active_node": FINAL_NODE,
        "pending_transfer": None,
        "is_final": True,
    }
