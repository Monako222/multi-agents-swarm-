from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from functools import lru_cache
from typing import Any, Final

from dotenv import dotenv_values
from loguru import logger

_DISABLED: Final = "SWARM_TRACING_DISABLED"

_SMITH_KEY: Final = "LANGSMITH_API_KEY"
_SMITH_KEY_ALT: Final = "LANGCHAIN_API_KEY"
_SMITH_URL: Final = "LANGSMITH_ENDPOINT"
_SMITH_URL_ALT: Final = "LANGCHAIN_ENDPOINT"
_SMITH_PROJECT: Final = "LANGSMITH_PROJECT"
_SMITH_PROJECT_ALT: Final = "LANGCHAIN_PROJECT"

_FUSE_PK: Final = "LANGFUSE_PUBLIC_KEY"
_FUSE_SK: Final = "LANGFUSE_SECRET_KEY"
_FUSE_URL: Final = "LANGFUSE_BASE_URL"
_FUSE_URL_ALT: Final = "LANGFUSE_HOST"

_TRACING_KEYS: Final[frozenset[str]] = frozenset(
    {
        _DISABLED,
        _SMITH_KEY,
        _SMITH_KEY_ALT,
        _SMITH_URL,
        _SMITH_URL_ALT,
        _SMITH_PROJECT,
        _SMITH_PROJECT_ALT,
        _FUSE_PK,
        _FUSE_SK,
        _FUSE_URL,
        _FUSE_URL_ALT,
    }
)


def _load_dotenv() -> None:
    values = dotenv_values(".env")
    for key in _TRACING_KEYS - os.environ.keys():
        if value := values.get(key):
            if clean := str(value).strip():
                os.environ[key] = clean


def _env(*names: str) -> str | None:
    for name in names:
        if value := os.getenv(name):
            if clean := value.strip():
                return clean
    return None


def _is_disabled() -> bool:
    return (_env(_DISABLED) or "").lower() in {"1", "true", "yes", "on"}


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def _patch_otel_detach() -> None:
    with suppress(ImportError):
        from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext

        if getattr(ContextVarsRuntimeContext.detach, "_swarm_safe", False):
            return

        original = ContextVarsRuntimeContext.detach

        def safe_detach(self: Any, token: object) -> None:
            with suppress(ValueError):
                original(self, token)

        safe_detach._swarm_safe = True  # type: ignore[attr-defined]
        ContextVarsRuntimeContext.detach = safe_detach


class TracingManager:
    __slots__ = ("_fuse", "_fuse_public_key", "_project", "_smith")

    def __init__(self, project: str = "TmpBot Swarm") -> None:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
        os.environ.setdefault("LANGSMITH_TRACING", "false")

        self._project = _env(_SMITH_PROJECT, _SMITH_PROJECT_ALT) or project
        self._smith = None if _is_disabled() else self._init_smith()
        self._fuse_public_key = _env(_FUSE_PK)
        self._fuse = None if _is_disabled() else self._init_fuse()

        active = [
            name
            for name, enabled in (
                ("LangSmith", self._smith is not None),
                ("Langfuse", self._fuse is not None),
            )
            if enabled
        ]
        logger.bind(tracing=active).info("Swarm tracing {}", "active" if active else "off")

    @property
    def enabled(self) -> bool:
        return self._smith is not None or self._fuse is not None

    def _init_smith(self) -> Any | None:
        api_key = _env(_SMITH_KEY, _SMITH_KEY_ALT)
        if not api_key:
            return None

        from langsmith import Client

        endpoint = _env(_SMITH_URL, _SMITH_URL_ALT)
        return Client(api_key=api_key, api_url=endpoint)

    def _init_fuse(self) -> Any | None:
        public_key = self._fuse_public_key
        secret_key = _env(_FUSE_SK)
        if not (public_key and secret_key):
            return None

        from langfuse import Langfuse

        base_url = _env(_FUSE_URL, _FUSE_URL_ALT)
        return Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url)

    def callbacks(self, *, tags: Sequence[str] = ()) -> list[Any]:
        callbacks: list[Any] = []

        if self._smith is not None:
            from langchain_core.tracers import LangChainTracer

            callbacks.append(
                LangChainTracer(
                    client=self._smith,
                    project_name=self._project,
                    tags=_unique(["swarm", *tags]),
                )
            )

        if self._fuse is not None:
            from langfuse.langchain import CallbackHandler

            callbacks.append(CallbackHandler(public_key=self._fuse_public_key))

        return callbacks

    def metadata(
        self,
        *,
        thread_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        extra: Mapping[str, Any] | None = None,
        tags: Sequence[str] = (),
    ) -> dict[str, Any]:
        meta = dict(extra or {})
        meta.update(
            {
                "service": "swarm-agent",
                "project": self._project,
                "thread_id": thread_id,
                "user_id": user_id,
                "session_id": session_id,
                "langfuse_user_id": user_id,
                "langfuse_session_id": session_id or thread_id,
                "langfuse_tags": _unique(["swarm", *tags]),
            }
        )
        return {key: value for key, value in meta.items() if value not in (None, "", [])}

    def runnable_config(
        self,
        *,
        base: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = dict(base or {})
        config["callbacks"] = [*list(config.get("callbacks") or ()), *self.callbacks(tags=tags)]
        config["tags"] = _unique([*list(config.get("tags") or ()), "swarm", *tags])
        config["metadata"] = {
            **dict(config.get("metadata") or {}),
            **self.metadata(
                thread_id=thread_id,
                user_id=user_id,
                session_id=session_id,
                extra=metadata,
                tags=tags,
            ),
        }

        configurable = dict(config.get("configurable") or {})
        if thread_id:
            configurable.setdefault("thread_id", thread_id)
        if configurable:
            config["configurable"] = configurable

        return config

    async def close(self) -> None:
        if self._fuse is None:
            return

        for method in ("flush", "shutdown"):
            func = getattr(self._fuse, method, None)
            if callable(func):
                with suppress(Exception):
                    result = func()
                    if hasattr(result, "__await__"):
                        await result
        self._fuse = None


@lru_cache(maxsize=1)
def get_tracing_manager(project: str = "TmpBot Swarm") -> TracingManager:
    return TracingManager(project=project)


_load_dotenv()
_patch_otel_detach()

__all__ = ["TracingManager", "get_tracing_manager"]
