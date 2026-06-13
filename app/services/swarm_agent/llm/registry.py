from typing import Self

from app.config import Settings
from app.services.swarm_agent.exceptions import RegistryValidationError
from ..types import (
    ProviderConfig,
    ModelProfile,
    ProviderName,
    Modality, 
)



class LLMRegistry:
    """In-memory реестр провайдеров 
    и профилей LLM. Гарантирует мгновенный 
    O(1) доступ и строгую валидацию 
    fallback-цепочек."""


    __slots__ = (
        "_profiles", 
        "_lookup",
        "_pvds",
    )


    def __init__(self) -> None:
        
        # Наши провайдеры 
        # (OpenRouter и т.д.)
        self._pvds: dict[
            ProviderName, 
            ProviderConfig
        ] = {}
        
        # Индекс для поиска 
        # как по alias, так и по model_id
        self._lookup: dict[str, ModelProfile] = {}
        
        # Изолированный реестр
        # уникальных профилей (ключ - alias)
        self._profiles: dict[str, ModelProfile] = {}
        
        
        
        
    def provider(
        self, 
        name: ProviderName
    ) -> ProviderConfig:
        """Мгновенно получаем
        конфиг провайдера."""
        
        try:
            return self._pvds[name]
        
        except KeyError as exc:
            raise KeyError(
                "Not found provider"
                f"{name}"
            ) from exc
     
            
            
    def model(
        self, 
        alias_id: str
    ) -> ModelProfile:
        """Мгновенно получаем 
        профиль модели."""
        
        try:
            return self._lookup[alias_id]
        
        except KeyError as exc:
            raise KeyError(
                "Not found model: "
                f"{alias_id}"
            ) from exc




    def by_modality(
        self, 
        modality: Modality
    ) -> tuple[ModelProfile, ...]:
        """Отдает список моделей,
        способных обработать нужный 
        тип данных."""
        
        return tuple(
            model for model 
            in self._profiles.values() 
            if model.supports(modality)
        )
        



    def reg_provider(self, pvd: ProviderConfig) -> Self:
        """Добавляет настройки 
        платформы в реестр."""
        
        if pvd.name in self._pvds: 
            raise ValueError(
                f"Provider {pvd.name!r} "
                "already registry."
            )

        self._pvds[pvd.name] = pvd
        return self




    def reg_model(self, model: ModelProfile) -> Self:
        """Индексирует модель 
        для роутинга."""
        
        if model.provider not in self._pvds:
            raise ValueError(
                f"Provider {model.provider!r} "
                "is missing in registry."
            )
            
        if model.alias in self._profiles:
            raise ValueError(
                f"Model {model.alias!r} "
                "already registry."
            )
            

        self._lookup[model.alias] = model
        self._profiles[model.alias] = model
        self._lookup.setdefault(model.model_id, model)
        
        return self




    def validate(self) -> None:
        """Проверяет целостность 
        графа моделей до первого
        сетевого вызова."""
        
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(alias: str) -> None:
            """Изолированный рекурсивный 
            поиск в глубину."""
            
            # 1. Если ветка уже была 
            # успешно отвалидирована ранее, 
            if alias in visited:
                return
                
            # 2. Если попали на узел,
            # который уже висит в стеке,
            if alias in visiting:
                raise RegistryValidationError(
                    f"Infinite fallback cycle "
                    f"detect at alias: {alias!r}"
                )
                

            visiting.add(alias)
            
            # 4. Итерируемся по 
            # всем запасным fallbacks
            for fallback in self._profiles[alias].fallbacks:
                
                # Защита от битых ссылок 
                # интегрирована прямо в обход.
                if fallback not in self._profiles:
                    raise RegistryValidationError(
                        f"Invalid fallback: {fallback!r} "
                        f"on model {alias!r}"
                    )
                    
                # Рекурсивно 
                # ныряем глубже
                dfs(fallback)
                
            # 5. Все пути корректны, 
            # заносим в белый список
            visiting.remove(alias)
            visited.add(alias)


        # 6. Стартуем алгоритм. 
        # Каждую модель проверяя 1 раз
        for alias in self._profiles:
            dfs(alias)









def get_llm_registry(
    secrets: Settings,
) -> LLMRegistry:
    """Собирает готовый к бою 
    реестр провайдеров и моделей 
    на базе провайдеров."""
    
    registry = LLMRegistry()
    
    # 1. Регистрация 
    # доступных провайдеров
    registry.reg_provider(
        ProviderConfig(
            name=ProviderName.OPENROUTER,
            api_key=secrets.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            headers={"HTTP-Referer": "https://agent",
                     "X-Title": "Swarm Agent"},
            connect_timeout_s=15.0,
            pool_timeout_s=15.0,
            timeout_s=120.0,
            max_retries=3,
            use_proxy=False,

        )
    )

    # 2. Поддерживающие
    # форматы моделями
    all_mods = frozenset({
        Modality.TEXT, 
        Modality.IMAGE, 
        Modality.AUDIO, 
        Modality.VIDEO,
        Modality.DOCS
    })
    
    vision_mods = frozenset({
        Modality.TEXT, 
        Modality.IMAGE
    })

    # 3. Декларативный 
    # список профилей
    profiles = (
        ModelProfile(
            alias="fast",
            provider=ProviderName.OPENROUTER,
            model_id="google/gemini-2.5-flash-lite",
            context_window=1_048_576,
            max_output_tokens=16_384,
            temperature=0.2,
            tags=frozenset({
                "fast", 
                "cheap",
            }),
            fallbacks=("fallback",),
            modalities=all_mods,
        ),
        ModelProfile(
            alias="reasoning",
            provider=ProviderName.OPENROUTER,
            model_id="openai/gpt-5-nano",
            context_window=256_000,
            max_output_tokens=None,
            temperature=0.1,
            tags=frozenset({
                "reasoning", 
                "final"
            }),
            fallbacks=("fallback",),
            modalities=vision_mods,
        ),
        ModelProfile(
            alias="fast_multimodal",
            provider=ProviderName.OPENROUTER,
            model_id="google/gemini-2.5-flash-lite",
            context_window=1_048_576,
            max_output_tokens=16_384,
            temperature=0.2,
            tags=frozenset({
                "fast",
                "multimodal",
            }),
            fallbacks=("fallback",),
            modalities=all_mods,
        ),
        ModelProfile(
            alias="fallback",
            provider=ProviderName.OPENROUTER,
            model_id="google/gemini-2.5-flash-lite",
            context_window=1_048_576,
            max_output_tokens=16_384,
            temperature=0.2,
            tags=frozenset({
                "fast", 
                "cheap",
            }),
            fallbacks=(),
            modalities=all_mods,
        ),
    )

    # 4. Мгновенная 
    # массовая регистрация
    for profile in profiles:
        registry.reg_model(
            model=profile
        )
        
    registry.validate()
    return registry
