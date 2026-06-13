"""Pydantic-схемы аргументов системных инструментов."""

from __future__ import annotations

from pydantic import Field

from app.services.swarm_agent.agents.tools.base import ArgsModel
from app.services.swarm_agent.types import (
    ErrorSeverity,
    FileKind,
    FinalText,
    JsonObject,
    LangCode,
    LongText,
    ShortText,
    TagText,
)


class FindingsArgs(ArgsModel):
    """Аргументы сохранения переиспользуемых фактов."""

    goal: ShortText | None = Field(default=None, description="Current step goal.")
    notes: list[ShortText] | None = Field(default=None, description="Compact reusable notes.")
    data: JsonObject | None = Field(default=None, description="Structured reusable JSON facts.")


class ContextArgs(ArgsModel):
    """Аргументы обновления нормализованного контекста."""

    query: ShortText | None = Field(default=None, description="Normalized user query.")
    intent: ShortText | None = Field(default=None, description="User intent.")
    subject: ShortText | None = Field(default=None, description="Request subject.")
    tags: list[TagText] | None = Field(default=None, description="Semantic routing tags.")
    lang: LangCode | None = Field(default=None, description="BCP-47-ish language code, e.g. ru.")


class ArtifactArgs(ArgsModel):
    """Аргументы регистрации выходного артефакта."""

    file_uri: LongText = Field(..., description="Canonical URI or path of the produced artifact.")
    description: ShortText = Field(..., description="Human-readable artifact description.")
    kind: FileKind = Field(default=FileKind.DOC, description="Artifact kind.")
    content_preview: LongText | None = Field(
        default=None,
        description="Optional short text preview.",
    )


class ErrorArgs(ArgsModel):
    """Аргументы записи recoverable ошибки."""

    message: LongText = Field(..., description="Diagnostic message.")
    severity: ErrorSeverity = Field(default=ErrorSeverity.ERROR, description="Severity level.")


class FinishArgs(ArgsModel):
    """Аргументы финализации всей задачи."""

    final: FinalText = Field(..., description="Final answer for the user.")
