"""Command-aware executor для tool calls.

Принимает OpenAI-совместимые `AIMessage.tool_calls`. Одно сообщение модели
может содержать сразу несколько вызовов. 
Безопасная стратегия batch-выполнения:
- Независимые non-routing инструменты выполняются параллельно.
- `finish` имеет безусловный приоритет над `transfer`.
- За один turn исполняется только один выбранный routing tool.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.types import Command
from loguru import logger

from app.services.swarm_agent.agents import FINAL_NODE, LOOP_GUARD_NODE
from app.services.swarm_agent.agents.tools.base import (
    is_parallel_safe,
    is_routing_tool,
    requires_tool_call_id,
)
from app.services.swarm_agent.exceptions import RegistryValidationError
from app.services.swarm_agent.graph.runtime.tool_protocol import (
    FINISH_TOOL,
    ToolCall,
    ToolOutcome,
    command_goto,
    command_update,
    extract_tool_call,
    has_tool_message,
    protocol_warning,
    strip_non_routing_state_changes,
    tool_message,
)
from app.services.swarm_agent.graph.state import ErrorRecord, SwarmState
from app.services.swarm_agent.types import ErrorSeverity
from app.services.swarm_agent.utils import as_list, merge_update, safe_error_text


class ToolExecutorNode:
    """Выполняет пакеты инструментов и маршрутизирует граф по Command.goto."""

    __slots__ = (
        "_caller_name",
        "_local_loop_limit",
        "_targets",
        "_tools",
    )

    def __init__(
        self,
        caller_name: str,
        tools: tuple[BaseTool, ...],
        *,
        valid_gotos: tuple[str, ...],
        local_loop_limit: int | None = None,
    ) -> None:
        
        self._caller_name = caller_name
        self._local_loop_limit = local_loop_limit
        self._tools = self._index_tools(caller_name, tools)
        self._targets = frozenset({caller_name, FINAL_NODE, *valid_gotos})

    @staticmethod
    def _index_tools(
        caller: str, 
        tools: tuple[BaseTool, ...]
    ) -> dict[str, BaseTool]:
        """Сборка индекса инструментов за O(N) с проверкой на дубликаты."""
        
        idx: dict[str, BaseTool] = {}
        dupes: set[str] = set()
        
        for t in tools:
            if t.name in idx:
                dupes.add(t.name)
            else:
                idx[t.name] = t
                
        if dupes:
            joined = ", ".join(sorted(dupes))
            raise RegistryValidationError(
                f"Duplicate tool names for {caller!r}: {joined}"
            )
            
        return idx

    async def __call__(
        self, 
        state: SwarmState, 
        config: RunnableConfig
    ) -> Command:
        """Молниеносный парсинг и запуск батча инструментов из AIMessage."""
        
        calls: list[ToolCall] = []
        
        # Извлекаем tool_calls за одно касание стейта через моржовые операторы
        if (msgs := state.get("messages")) and (raw := getattr(msgs[-1], "tool_calls", None)):
            calls = [extract_tool_call(c, index=i) for i, c in enumerate(raw)]

        if not calls:
            return Command(
                update={
                    "errors": [
                        ErrorRecord(
                            source=self._caller_name,
                            severity=ErrorSeverity.WARNING,
                            message="Tool executor reached without tool calls.",
                        )
                    ]
                },
                goto=self._caller_name,
            )

        # Выбираем ветку выполнения в зависимости от лимитов графа
        if self._limit_reached(state):
            outcomes = await self._execute_at_limit(calls, config)
        else:
            outcomes = await self._execute_calls(self._cap_calls(calls), config)
            
        return self._merge_outcomes(outcomes)

    def _merge_outcomes(self, outcomes: list[ToolOutcome]) -> Command:
        """Слияние результатов разрозненных вызовов в единый Command."""
        
        update: dict[str, Any] = {}
        goto = self._caller_name
        
        for outcome in sorted(outcomes, key=lambda item: item.index):
            merge_update(update, outcome.update)
            if outcome.routes:
                goto = outcome.goto
                
        return Command(update=update, goto=goto or self._caller_name)

    def _limit_reached(self, state: SwarmState) -> bool:
        """Сверхбыстрая проверка бизнес-лимитов."""
        
        if int(state.get("total_steps") or 0) >= 64:
            return True
            
        max_loops = self._local_loop_limit or 8
        return int(state.get("loops") or 0) >= max_loops

    def _cap_calls(self, calls: list[ToolCall]) -> list[ToolCall]:
        """Обрезает излишний fan-out (DDoS провайдера), сохраняя routing tool."""
        
        limit = 8
        if len(calls) <= limit:
            return calls

        selected = self._selected_routing_call(calls)
        sel_idx = selected.index if selected else -1
        
        kept: list[ToolCall] = []
        for c in calls:
            if len(kept) >= limit:
                break
            if c.index != sel_idx:
                kept.append(c)

        # Гарантируем, что routing-инструмент попадет в лимитированный список
        if selected and selected not in kept:
            if len(kept) >= limit:
                kept[-1] = selected
            else:
                kept.append(selected)

        # Возвращаем исполняемые + обрезанные (для protocol warning)
        kept_ids = {c.index for c in kept}
        capped = sorted(kept, key=lambda c: c.index)
        capped.extend(c for c in calls if c.index not in kept_ids)
        
        return capped

    async def _execute_at_limit(
        self,
        calls: list[ToolCall],
        config: RunnableConfig,
    ) -> list[ToolOutcome]:
        """Экстренное завершение: исполняется только finish(), остальное отсекается."""
        
        if finish := next((c for c in calls if c.name == FINISH_TOOL), None):
            return await self._execute_finish_at_limit(calls, finish, config)

        msg_content = (
            "Ignored because loop/step limit was reached; "
            "loop guard will finish."
        )
        err_msg = "Tool call skipped at loop/step limit before loop guard."

        return [
            ToolOutcome(
                index=c.index,
                update={
                    "messages": [
                        tool_message(
                            msg_content, 
                            name=c.name, 
                            call_id=c.call_id
                        )
                    ],
                    "errors": [
                        ErrorRecord(
                            source=self._caller_name,
                            severity=ErrorSeverity.WARNING,
                            message=err_msg,
                        )
                    ],
                },
                goto=LOOP_GUARD_NODE,
                routes=True,
            )
            for c in calls
        ]

    async def _execute_finish_at_limit(
        self,
        calls: list[ToolCall],
        finish: ToolCall,
        config: RunnableConfig,
    ) -> list[ToolOutcome]:
        """Отработка finish на лимите шагов."""
        
        outcomes = [
            protocol_warning(
                c, 
                "Ignored because loop/step limit was already reached."
            )
            for c in calls if c.index != finish.index
        ]
        
        if tool := self._tools.get(finish.name):
            outcomes.append(
                await self._run_tool(finish, tool, config, routes=True)
            )
        else:
            outcomes.append(self._unknown_tool(finish))
            
        return outcomes

    async def _execute_calls(
        self,
        calls: list[ToolCall],
        config: RunnableConfig,
    ) -> list[ToolOutcome]:
        """Диспетчер вызовов: сортирует батч на параллельные и последовательные."""
        
        selected = self._selected_routing_call(calls)
        barrier = selected.index if selected else None
        
        exec_ids = {c.index for c in calls[:8]}
        if selected:
            exec_ids.add(selected.index)

        outcomes: list[ToolOutcome] = []
        parallel: list[tuple[ToolCall, BaseTool]] = []
        sequential: list[tuple[ToolCall, BaseTool]] = []
        routing: list[tuple[ToolCall, BaseTool]] = []

        for c in calls:
            bucket = self._classify_call(c, selected, barrier, exec_ids)
            if isinstance(bucket, ToolOutcome):
                outcomes.append(bucket)
            elif bucket[0] == "parallel":
                parallel.append((c, bucket[1]))
            elif bucket[0] == "sequential":
                sequential.append((c, bucket[1]))
            else:
                routing.append((c, bucket[1]))

        # Мощный I/O параллелизм для non-routing вызовов
        outcomes.extend(await self._run_parallel(parallel, config))
        
        for c, t in sequential:
            outcomes.append(await self._run_tool(c, t, config, routes=False))
            
        for c, t in routing:
            outcomes.append(await self._run_tool(c, t, config, routes=True))
            
        return outcomes

    def _classify_call(
        self,
        call: ToolCall,
        selected: ToolCall | None,
        barrier: int | None,
        exec_ids: set[int],
    ) -> ToolOutcome | tuple[str, BaseTool]:
        """Строгий классификатор безопасности вызова."""
        
        if call.index not in exec_ids:
            return protocol_warning(call, "Ignored because tool fan-out was capped.")

        if not (tool := self._tools.get(call.name)):
            return self._unknown_or_ignored_unknown(call, barrier=barrier)

        if selected and call.index == selected.index:
            return "routing", tool
            
        if selected and call.index > selected.index:
            return protocol_warning(
                call,
                "Ignored because it was placed after a routing tool in the same turn.",
            )
            
        if is_routing_tool(tool):
            return protocol_warning(
                call,
                "Ignored because another routing tool was selected for this turn.",
            )
            
        return ("parallel", tool) if is_parallel_safe(tool) else ("sequential", tool)

    async def _run_parallel(
        self,
        calls: Sequence[tuple[ToolCall, BaseTool]],
        config: RunnableConfig,
    ) -> list[ToolOutcome]:
        """Выполняет parallel-safe инструменты батчами для защиты от rate-limits."""
        
        if not calls:
            return []
            
        out: list[ToolOutcome] = []
        chunk_size = 4
        
        for i in range(0, len(calls), chunk_size):
            chunk = calls[i : i + chunk_size]
            out.extend(
                await asyncio.gather(
                    *(self._run_tool(c, t, config, routes=False) for c, t in chunk)
                )
            )
            
        return out

    def _selected_routing_call(self, calls: list[ToolCall]) -> ToolCall | None:
        """Мгновенно находит приоритетный routing-инструмент (finish > transfer)."""
        
        routings = [
            c for c in calls 
            if (t := self._tools.get(c.name)) and is_routing_tool(t)
        ]
        
        if not routings:
            return None
            
        if finishes := [c for c in routings if c.name == FINISH_TOOL]:
            return min(finishes, key=lambda c: c.index)
            
        return min(routings, key=lambda c: c.index)

    def _unknown_or_ignored_unknown(
        self, 
        call: ToolCall, 
        *, 
        barrier: int | None
    ) -> ToolOutcome:
        """Обрабатывает неизвестные инструменты с учетом маршрутизационного барьера."""
        
        if barrier is not None and call.index > barrier:
            return protocol_warning(
                call,
                "Ignored unknown tool because it was placed after a routing tool.",
            )
            
        return self._unknown_tool(call)

    def _unknown_tool(self, call: ToolCall) -> ToolOutcome:
        """Формирует безопасный update при попытке LLM выдумать инструмент."""
        
        logger.bind(agent=self._caller_name, tool=call.name).warning("Unknown tool")
        
        msg = tool_message(
            f"Unknown tool: {call.name!r}",
            name=call.name,
            call_id=call.call_id,
        )
        err = ErrorRecord(
            source=self._caller_name,
            severity=ErrorSeverity.ERROR,
            message=f"Model requested unknown tool: {call.name!r}.",
        )
        
        return ToolOutcome(
            call.index, 
            {"messages": [msg], "errors": [err]}, 
            self._caller_name
        )

    async def _run_tool(
        self,
        call: ToolCall,
        tool: BaseTool,
        config: RunnableConfig,
        *,
        routes: bool,
    ) -> ToolOutcome:
        """Обертка над вызовом `ainvoke` инструмента с перехватом падений."""
        
        try:
            result = await tool.ainvoke(self._tool_input(call, tool), config=config)
        except Exception as exc:  # noqa: BLE001
            return self._failed_tool_outcome(call, exc)

        if isinstance(result, Command):
            return self._command_outcome(call, result, routes=routes)

        msg = tool_message(
            str(result), 
            name=call.name, 
            call_id=call.call_id
        )
        
        return ToolOutcome(call.index, {"messages": [msg]}, self._caller_name)

    @staticmethod
    def _tool_input(call: ToolCall, tool: BaseTool) -> dict[str, Any]:
        """Подготовить input для tool и явно передать tool_call_id."""

        args = dict(call.args)

        if requires_tool_call_id(tool):
            args["tool_call_id"] = call.call_id

        return args

    def _failed_tool_outcome(
        self, 
        call: ToolCall, 
        exc: BaseException
    ) -> ToolOutcome:
        """Конвертирует Python-исключение инструмента в текстовый ответ для LLM."""
        
        err_text = safe_error_text(exc)
        logger.bind(agent=self._caller_name, tool=call.name).exception("Tool failed")
        
        msg = tool_message(
            f"Tool failed: {err_text}",
            name=call.name,
            call_id=call.call_id,
        )
        err = ErrorRecord(
            source=self._caller_name,
            severity=ErrorSeverity.ERROR,
            message=f"Tool {call.name!r} failed: {err_text}",
        )
        
        return ToolOutcome(
            call.index, 
            {"messages": [msg], "errors": [err]}, 
            self._caller_name
        )

    def _command_outcome(
        self, 
        call: ToolCall, 
        command: Command, 
        *, 
        routes: bool
    ) -> ToolOutcome:
        """Парсинг Command с жесткой защитой от изменения управляющего стейта."""
        
        update = command_update(command)
        goto = command_goto(command, self._caller_name)

        # Жесткая валидация узла назначения
        if goto not in self._targets:
            goto, routes = self._handle_invalid_goto(call, update, goto)

        # Если инструмент не маршрутизирующий — блокируем смену графа
        if not routes:
            goto = self._sanitize_non_routing_command(call, update, goto)

        # Закрываем протокол: инструмент обязан вернуть ToolMessage
        if not has_tool_message(update, call.call_id):
            update["messages"] = [
                *as_list(update.get("messages")),
                tool_message("ok", name=call.name, call_id=call.call_id),
            ]
            
        return ToolOutcome(call.index, update, goto, routes=routes)

    def _handle_invalid_goto(
        self,
        call: ToolCall,
        update: dict[str, Any],
        goto: str,
    ) -> tuple[str, bool]:
        """Сброс нелегитимного goto с записью системной ошибки."""
        
        logger.bind(agent=self._caller_name, tool=call.name, goto=goto).error(
            "Invalid tool goto"
        )
        err = ErrorRecord(
            source=self._caller_name,
            severity=ErrorSeverity.ERROR,
            message=f"Tool {call.name!r} returned invalid goto: {goto!r}.",
        )
        update["errors"] = [*as_list(update.get("errors")), err]
        
        return self._caller_name, False

    def _sanitize_non_routing_command(
        self,
        call: ToolCall,
        update: dict[str, Any],
        goto: str,
    ) -> str:
        """Зачистка попыток обычного инструмента вмешаться в маршрутизацию."""
        
        if goto != self._caller_name:
            err = ErrorRecord(
                source=self._caller_name,
                severity=ErrorSeverity.ERROR,
                message=f"Non-routing tool {call.name!r} tried to route to {goto!r}.",
            )
            update["errors"] = [*as_list(update.get("errors")), err]
            
        strip_non_routing_state_changes(
            update,
            caller_name=self._caller_name,
            tool_name=call.name,
        )
        
        return self._caller_name
