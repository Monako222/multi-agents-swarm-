"""Тонкий async-клиент OpenRouter web-search server tool."""

from __future__ import annotations

import asyncio
from os import getenv
from typing import Any
from typing import Final
from hashlib import blake2b
from collections.abc import Mapping
from collections.abc import Sequence

import httpx
from pydantic import Field
from pydantic import BaseModel
from pydantic import ConfigDict

from app.config import Settings
from app.config import get_settings
from app.services.swarm_agent.utils import clip
from app.services.swarm_agent.types import LinkRecord
from app.services.swarm_agent.exceptions import MissingApiKeyError
from app.services.swarm_agent.exceptions import ToolExecutionError


_SOURCE_NAME: Final[str] = "openrouter:web_search"


def _stable_id(prefix: str, value: str) -> str:
    digest = blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return f"{prefix}:{digest}"


class WebSearchSource(BaseModel):
    """Нормализованный внешний источник из url_citation annotation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    url: str
    title: str = ""
    snippet: str = ""
    source: str = _SOURCE_NAME

    def to_link_record(self) -> LinkRecord:
        return LinkRecord(
            id=self.id,
            url=self.url,
            title=self.title,
            snippet=self.snippet,
            source=self.source,
        )

    def to_json(self) -> dict[str, str]:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source,
        }


class WebSearchResult(BaseModel):
    """Нормализованный результат одного OpenRouter web-search запроса."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    query: str
    answer: str = ""
    sources: tuple[WebSearchSource, ...] = Field(default_factory=tuple)

    def to_tool_json(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [src.to_json() for src in self.sources],
        }

    def to_state_json(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [src.to_json() for src in self.sources],
        }


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return value
    return ()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    parts: list[str] = []
    for part in _as_sequence(content):
        if isinstance(part, str):
            parts.append(part)
            continue

        mapping = _as_mapping(part)
        if isinstance(text := mapping.get("text"), str):
            parts.append(text)

    return "\n".join(p for p in parts if p).strip()


def _parse_sources(annotations: Any) -> tuple[WebSearchSource, ...]:
    sources: list[WebSearchSource] = []
    seen: set[str] = set()

    for raw in _as_sequence(annotations):
        annotation = _as_mapping(raw)
        if annotation.get("type") != "url_citation":
            continue

        citation = _as_mapping(annotation.get("url_citation"))
        url = str(citation.get("url") or "").strip()
        if not url or url in seen:
            continue

        seen.add(url)
        sources.append(
            WebSearchSource(
                id=_stable_id("web", url),
                url=url,
                title=str(citation.get("title") or "").strip(),
                snippet=clip(
                    str(citation.get("content") or "").strip(),
                    1_200,
                ),
            )
        )

    return tuple(sources)


def parse_openrouter_response(
    query: str, payload: Mapping[str, Any]
) -> WebSearchResult:
    """Извлечь answer и citations из OpenRouter chat completion."""

    first_choice = next(iter(_as_sequence(payload.get("choices"))), {})
    message = _as_mapping(_as_mapping(first_choice).get("message"))

    return WebSearchResult(
        id=_stable_id("web_query", query),
        query=query,
        answer=_message_text(message.get("content")),
        sources=_parse_sources(message.get("annotations")),
    )


def build_web_search_payload(
    *,
    query: str,
    allowed_domains: Sequence[str] | None,
    excluded_domains: Sequence[str] | None,
    max_results: int,
    search_context_size: str,
) -> dict[str, Any]:
    """Собрать OpenRouter payload с server tool openrouter:web_search."""

    parameters: dict[str, Any] = {
        "engine": "auto",
        "max_results": max_results,
        "max_total_results": max_results,
        "search_context_size": search_context_size,
    }

    if allowed_domains:
        parameters["allowed_domains"] = list(allowed_domains)
    if excluded_domains:
        parameters["excluded_domains"] = list(excluded_domains)

    return {
        "model": "google/gemini-2.5-flash-lite",
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer the user's search query using current web data when useful. "
                    "Be concise and preserve citation annotations returned by the provider."
                ),
            },
            {"role": "user", "content": query},
        ],
        "tools": [
            {
                "type": "openrouter:web_search",
                "parameters": parameters,
            }
        ],
    }


class OpenRouterWebSearchClient:
    """Минимальный OpenRouter client для web-search tool calls."""

    __slots__ = ("_client", "_secrets")

    def __init__(
        self,
        *,
        secrets: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._secrets = secrets or get_settings()
        self._client = http_client

    async def search(
        self,
        *,
        query: str,
        allowed_domains: Sequence[str] | None,
        excluded_domains: Sequence[str] | None,
        max_results: int,
        search_context_size: str,
    ) -> WebSearchResult:
        payload = build_web_search_payload(
            query=query,
            allowed_domains=allowed_domains,
            excluded_domains=excluded_domains,
            max_results=max_results,
            search_context_size=search_context_size,
        )
        response = await self._post(payload)
        return parse_openrouter_response(query, response)

    async def search_batch(
        self,
        *,
        queries: Sequence[str],
        allowed_domains: Sequence[str] | None,
        excluded_domains: Sequence[str] | None,
        max_results: int,
        search_context_size: str,
    ) -> list[WebSearchResult]:
        if not queries:
            return []

        return list(
            await asyncio.gather(
                *(
                    self.search(
                        query=query,
                        allowed_domains=allowed_domains,
                        excluded_domains=excluded_domains,
                        max_results=max_results,
                        search_context_size=search_context_size,
                    )
                    for query in queries
                )
            )
        )

    async def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._client is not None:
            return await self._post_with(self._client, payload)

        kwargs: dict[str, Any] = {
            "timeout": self._timeout(),
            "http2": True,
        }
        if proxy_url := getenv("HTTPS_PROXY") or getenv("HTTP_PROXY"):
            kwargs["proxy"] = proxy_url

        async with httpx.AsyncClient(**kwargs) as client:
            return await self._post_with(client, payload)

    async def _post_with(
        self,
        client: httpx.AsyncClient,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        response = await client.post(
            self._chat_url(),
            headers=self._headers(),
            json=payload,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = clip(exc.response.text, 600)
            raise ToolExecutionError(
                f"OpenRouter web search failed: {exc.response.status_code} {detail}"
            ) from exc

        parsed = response.json()
        if not isinstance(parsed, Mapping):
            raise ToolExecutionError("OpenRouter web search returned non-object JSON.")

        return {str(key): value for key, value in parsed.items()}

    def _chat_url(self) -> str:
        return "https://openrouter.ai/api/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        api_key = self._secrets.OPENROUTER_API_KEY
        if api_key is None:
            raise MissingApiKeyError("OPENROUTER_API_KEY is missing.")

        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://agent",
            "X-Title": "Swarm Agent",
        }

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            timeout=120.0,
            connect=15.0,
            pool=15.0,
            write=120.0,
            read=120.0,
        )
