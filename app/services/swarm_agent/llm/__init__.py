import httpx

from typing import Any
from threading import RLock

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel


from app.services.swarm_agent.config import Settings, get_settings
from app.services.swarm_agent.exceptions import MissingApiKeyError


from .registry import LLMRegistry, get_llm_registry
from .http import AsyncHTTPClientPool
from ..types import ModelProfile



class LLMHub:
    """Thread-safe кэширующая фабрика 
    ChatOpenAI-моделей. Гарантирует сборку 
    каждой модели ровно один раз и 
    моментальную выдачу из кэша.
    """

    __slots__ = (
        "registry",
        "_cache", 
        "_lock", 
        "pool"
    )


    def __init__(
        self,
        settings: Settings
    ) -> None:

        self._lock = RLock()
        self.registry = get_llm_registry(settings)
        self.pool = AsyncHTTPClientPool(settings.proxy_url)
        self._cache: dict[str, BaseChatModel] = {}
        


    def _build(self, profile: ModelProfile) -> BaseChatModel:
        """Внутренняя мгновенная сборка 
        LangChain-инстанса без I/O."""
        
        
        provider = self.registry.provider(
            profile.provider
        )
        
        if not provider.api_key:
            raise MissingApiKeyError(
                "API key for provider "
                f"{provider.name!s} is missing."
            )


        # Полностью декларативная 
        # сборка ядра конфигурации
        kwargs: dict[str, Any] = {
            "model": profile.model_id,
            "api_key": provider.api_key,
            "base_url": provider.base_url,
            "temperature": profile.temperature,
            "max_retries": provider.max_retries,
            "default_headers": dict(provider.headers) or None,
            "http_async_client": self.pool.client(provider.use_proxy),
            "timeout": httpx.Timeout(
                timeout=provider.timeout_s,
                connect=provider.connect_timeout_s,
                pool=provider.pool_timeout_s,
                write=provider.timeout_s,
                read=provider.timeout_s
            ),
        }


        # Расширяем инстанст
        # опциональными параметрами 
        tokens = profile.max_output_tokens
        if tokens:
            kwargs["max_tokens"] = tokens
            
        extra = profile.extra_body
        if extra:
            kwargs["extra_body"] = dict(extra)
        
        
        return ChatOpenAI(**kwargs).with_config(
            tags=sorted(profile.tags),
            metadata={"model_alias": profile.alias, 
                      "provider": str(profile.provider)},
        )





    def get(self, alias_or_id: str) -> BaseChatModel:
        """O(1) доступ к кэшу через Double-Checked Locking."""
        
        model = self._cache.get(
            alias_or_id
        )
        if model: 
            return model

        with self._lock:
            if alias_or_id in self._cache:
                return self._cache[alias_or_id]

            profile = self.registry.model(alias_or_id)
            model = self._build(profile)
            
            if profile.fallbacks:
                backups = [self.get(alias) for alias in profile.fallbacks]
                model = model.with_fallbacks(backups)  

            self._cache[profile.alias] = model
            self._cache[profile.model_id] = model
            
            return model



    async def aclose(self) -> None:
        await self.pool.aclose()


class LazyLLMHub:
    """Ленивая обертка, чтобы сборка графа не требовала API key до первого LLM-вызова."""

    __slots__ = ("_hub", "_lock", "_settings")

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = RLock()
        self._hub: LLMHub | None = None

    def get(self) -> LLMHub:
        if self._hub is not None:
            return self._hub
        with self._lock:
            if self._hub is None:
                self._hub = LLMHub(self._settings)
            return self._hub

    async def aclose(self) -> None:
        if self._hub is not None:
            await self._hub.aclose()
        
        
        
        
        
        
__all__ = [
    "LLMHub",
    "LazyLLMHub",
    "LLMRegistry"
    ]
