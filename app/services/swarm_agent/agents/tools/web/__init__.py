"""Factory web-инструментов для агента актуальной информации."""

from __future__ import annotations

from typing import Annotated, Any

import orjson
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from app.services.swarm_agent.agents.tools.base import make_async_tool
from app.services.swarm_agent.config import Settings, get_settings
from app.services.swarm_agent.text import truncate_text
from app.services.swarm_agent.types import ToolCategory

from .client import OpenRouterWebSearchClient, WebSearchResult
from .schemas import SearchContextSize, WebSearchBatchArgs


def _compact_json(value: Any) -> str:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def _tool_message(content: str, *, call_id: str) -> ToolMessage:
    kwargs: dict[str, Any] = {
        "content": truncate_text(content, 4_000),
        "name": "web_search_batch",
        "tool_call_id": call_id,
        "id": f"tool:{call_id}",
    }

    try:
        return ToolMessage(**kwargs)
    except TypeError:
        kwargs.pop("id", None)
        return ToolMessage(**kwargs)


def _limit_domains(
    domains: list[str] | None,
    *,
    settings: Settings,
) -> list[str] | None:
    if not domains:
        return None
    return domains[: settings.web_search_max_domains]


def _tool_payload(results: list[WebSearchResult]) -> dict[str, Any]:
    return {
        "results": [result.to_tool_json() for result in results],
        "search_requests": sum(result.search_requests for result in results),
    }


def _state_json(results: list[WebSearchResult]) -> dict[str, Any]:
    return {
        "web_search": {
            result.id: result.to_state_json()
            for result in results
        }
    }


def build_web_tools(
    caller: str,
    client: OpenRouterWebSearchClient | None = None,
) -> list[BaseTool]:
    """Собрать web tools для конкретного agent node."""

    settings = get_settings()
    web_client = client or OpenRouterWebSearchClient(settings=settings)

    async def web_search_batch(
        queries: list[str],
        tool_call_id: str = "",
        allowed_domains: list[str] | None = None,
        excluded_domains: list[str] | None = None,
        max_results: int = 5,
        search_context_size: SearchContextSize = "medium",
    ) -> Command:
        """Выполнить несколько независимых web-search запросов одним tool call."""

        if not tool_call_id:
            tool_call_id = "web_search_batch_manual"

        bounded_queries = queries[: settings.web_search_max_queries]
        bounded_results = min(max_results, settings.web_search_max_results)
        context_size = search_context_size or settings.web_search_context_size

        results = await web_client.search_batch(
            queries=bounded_queries,
            allowed_domains=_limit_domains(allowed_domains, settings=settings),
            excluded_domains=_limit_domains(excluded_domains, settings=settings),
            max_results=bounded_results,
            search_context_size=context_size,
        )
        payload = _tool_payload(results)
        links = [
            source.to_link_record()
            for result in results
            for source in result.sources
        ]

        return Command(
            update={
                "messages": [
                    _tool_message(
                        _compact_json(payload),
                        call_id=tool_call_id,
                    )
                ],
                "data": {
                    "peers": links,
                    "json_data": _state_json(results),
                },
                "metrics": {
                    "tool_calls": 1,
                    "llm_calls": len(results),
                    "web_search_requests": payload["search_requests"],
                },
            },
            goto=caller,
        )

    return [
        make_async_tool(
            coroutine=web_search_batch,
            name="web_search_batch",
            description=(
                "Search current web data for one or more independent queries. "
                "Returns concise answers and cited sources."
            ),
            args_schema=WebSearchBatchArgs,
            category=ToolCategory.WEB,
            parallel_safe=True,
            routes=False,
            requires_tool_call_id=True,
        )
    ]


__all__ = [
    "OpenRouterWebSearchClient",
    "WebSearchBatchArgs",
    "WebSearchResult",
    "build_web_tools",
]
