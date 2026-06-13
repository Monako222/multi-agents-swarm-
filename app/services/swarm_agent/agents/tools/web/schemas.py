"""Pydantic-схемы аргументов web-инструментов."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator

from app.services.swarm_agent.agents.tools.base import ArgsModel
from app.services.swarm_agent.types import ShortText


_DOMAIN_RX = r"^(?:\*\.)?[A-Za-z0-9][A-Za-z0-9.-]*(?:/[^\s]*)?$"

type DomainFilter = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=253,
        pattern=_DOMAIN_RX,
    ),
]

type SearchContextSize = Literal["low", "medium", "high"]


class WebSearchBatchArgs(ArgsModel):
    """Аргументы batch web-search tool."""

    queries: list[ShortText] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Independent web-search queries.",
    )
    tool_call_id: str = Field(
        default="",
        exclude=True,
        description="Internal tool call id injected by ToolExecutorNode.",
    )
    allowed_domains: list[DomainFilter] | None = Field(
        default=None,
        max_length=10,
        description=(
            "Optional domain/path allowlist, "
            "e.g. arxiv.org or example.com/blog."
        ),
    )
    excluded_domains: list[DomainFilter] | None = Field(
        default=None,
        max_length=10,
        description="Optional domain/path denylist.",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum web results per query.",
    )
    search_context_size: SearchContextSize = Field(
        default="medium",
        description="Search context budget: low, medium, or high.",
    )

    @field_validator("queries")
    @classmethod
    def _dedupe_queries(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @field_validator("allowed_domains", "excluded_domains")
    @classmethod
    def _clean_domains(cls, value: list[str] | None) -> list[str] | None:
        if not value:
            return None

        cleaned: list[str] = []
        for domain in dict.fromkeys(value):
            normalized = domain.strip()

            if normalized.startswith(("http://", "https://")):
                raise ValueError(
                    "Domain filters must not include URL scheme. "
                    "Use example.com instead of https://example.com."
                )

            cleaned.append(normalized)

        return cleaned