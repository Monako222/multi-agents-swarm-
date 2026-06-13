"""Реестр топологии агентов и подагентов.

Реестр исключительно валидирует дерево агентов, строит быстрые lookup индексы 
и гарантирует, что transfer уйдет только в разрешённый узел.
В рантайме отсутствует поиск O(N) по спискам — только мгновенный O(1) доступ.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Final

from app.services.swarm_agent.exceptions import RegistryValidationError
from app.services.swarm_agent.types import AgentSpec, ToolCategory

ENTRY_NODE: Final[str] = "reasoner"
FINAL_NODE: Final[str] = "finalizer"
RECOVERY_NODE: Final[str] = "recover_unstructured_answer"
LOOP_GUARD_NODE: Final[str] = "loop_guard"
INIT_NODE: Final[str] = "init"

_RESERVED: Final[frozenset[str]] = frozenset({
    FINAL_NODE, 
    RECOVERY_NODE, 
    LOOP_GUARD_NODE, 
    INIT_NODE
})

_KNOWN_TOOLS: Final[frozenset[str]] = frozenset(
    c.value for c in ToolCategory
)

DEFAULT_AGENTS: Final[tuple[AgentSpec, ...]] = (
    AgentSpec(
        name="reasoner",
        role="Главный координатор: понять задачу, выбрать специалистов, собрать ответ.",
        tasks=(
            "Определить цель пользователя, ограничения и критерии результата.",
            "Решить, можно ли ответить сразу, или требуется профильный агент.",
            "Синтезировать findings специалистов в точный финальный ответ.",
        ),
        # peers=("vision", "audio", "video", "docs", "web"),
        peers=("web",),
        rules=(
            "Делегируй задачу только при явной пользе. Если ответ очевиден "
            "из контекста, сразу вызывай finish. После ответа специалиста "
            "обязательно верни управление себе или заверши задачу."
        ),
        model_alias="reasoning",
    ),
    # AgentSpec(
    #     name="vision",
    #     role="Аналитик изображений, скриншотов, схем, графиков и интерфейсов.",
    #     tasks=(
    #         "Извлечь визуальные факты, текст, структуру UI, элементы схем.",
    #         "Сжать наблюдения в компактные findings для reasoner/docs/video.",
    #     ),
    #     peers=("docs", "video", "web", "reasoner"),
    #     rules=(
    #         "Фиксируй исключительно наблюдаемые визуальные факты. "
    #         "Запрещено выдумывать OCR для неразборчивого текста."
    #     ),
    #     model_alias="reasoning",
    # ),
    # AgentSpec(
    #     name="audio",
    #     role="Аналитик аудио: речь, события, тональность, шумы и факты.",
    #     tasks=(
    #         "Выделить речь, события, участников и факты из аудио.",
    #         "Передать результат reasoner или video для синхронизации.",
    #     ),
    #     peers=("video", "web", "reasoner"),
    #     rules=(
    #         "Запрещено выдавать транскрипт как точный, если качество звука "
    #         "или источник вызывают сомнения."
    #     ),
    #     model_alias="reasoning",
    # ),
    # AgentSpec(
    #     name="video",
    #     role="Аналитик видео: сцены, таймкоды, действия, связь аудио/видео.",
    #     tasks=(
    #         "Разобрать видео на события, сцены и важные таймкоды.",
    #         "Связать визуальные и аудио-факты в пригодный summary.",
    #     ),
    #     peers=("vision", "audio", "docs", "web", "reasoner"),
    #     rules=(
    #         "Делай выводы исключительно по предоставленным кадрам."
    #     ),
    #     model_alias="reasoning",
    # ),
    # AgentSpec(
    #     name="docs",
    #     role="Аналитик документов: PDF, DOCX, таблицы, цитаты.",
    #     tasks=(
    #         "Извлечь факты, таблицы, фрагменты и структуру файлов.",
    #         "Сохранить findings для моментального доступа reasoner.",
    #     ),
    #     peers=("vision", "video", "web", "reasoner"),
    #     rules=(
    #         "Разделяй дословные цитаты, краткие выводы и предположения. "
    #         "Запрещено выдумывать ссылки на файлы."
    #     ),
    #     model_alias="reasoning",
    # ),
    AgentSpec(
        name="web",
        role="Исследователь актуальной внешней информации.",
        tasks=(
            "Определить, требуется ли свежий поиск или достаточно контекста.",
            "Сохранить проверенные источники и краткий синтез.",
        ),
        peers=("reasoner",),
        tools=(ToolCategory.WEB.value,),
        rules=(
            "Сообщай о поиске исключительно после выполнения web-tool. "
            "Предпочитай batch-инструменты для независимых запросов."
        ),
        model_alias="fast",
    ),
)


def walk_specs(
    specs: Sequence[AgentSpec] = DEFAULT_AGENTS,
    *,
    parent: str | None = None,
) -> Iterator[tuple[AgentSpec, str | None]]:
    """Плоский рекурсивный обход дерева specs с parent-ссылкой."""
    
    for spec in specs:
        yield spec, parent
        
        if spec.children:
            yield from walk_specs(spec.children, parent=spec.name)


class AgentRegistry:
    """Иммутабельный валидированный реестр агентов.
    
    Индексы строятся единожды при старте. Гарантирует O(1) доступ 
    и абсолютную безопасность передачи управления в рое.
    """

    __slots__ = (
        "_agents",
        "_children",
        "_parents",
        "_peer_names",
        "_peer_roles",
        "entry_node",
    )

    def __init__(
        self, 
        specs: Sequence[AgentSpec], 
        *, 
        entry_node: str = ENTRY_NODE
    ) -> None:
        
        agents: dict[str, AgentSpec] = {}
        parents: dict[str, str | None] = {}
        children: dict[str, list[str]] = {}

        # 1. Плоский обход и строгая валидация базовых правил
        for spec, parent in walk_specs(tuple(specs)):
            name = spec.name

            if name in agents:
                raise RegistryValidationError(f"Duplicate agent: {name!r}")
                
            if name in _RESERVED:
                raise RegistryValidationError(f"Reserved name: {name!r}")
                
            if unknown := set(spec.tools) - _KNOWN_TOOLS:
                joined = ", ".join(sorted(unknown))
                raise RegistryValidationError(
                    f"Agent {name!r} unknown tools: {joined}"
                )

            agents[name] = spec
            parents[name] = parent
            children.setdefault(name, [])
            
            if parent:
                children.setdefault(parent, []).append(name)

        if entry_node not in agents:
            raise RegistryValidationError(f"Missing entry node: {entry_node!r}")

        roles = {n: s.role for n, s in agents.items()}
        all_names = set(agents)

        peer_names: dict[str, tuple[str, ...]] = {}
        peer_roles: dict[str, Mapping[str, str]] = {}

        # 2. Построение матрицы разрешенных соседей (peers)
        for name, spec in agents.items():
            peers = set(spec.peers)
            
            if p := parents[name]:
                peers.add(p)
                
            peers.update(children.get(name, ()))

            if name in peers:
                raise RegistryValidationError(
                    f"Agent {name!r} self-referenced in peers."
                )

            if unknown := peers - all_names:
                joined = ", ".join(sorted(unknown))
                raise RegistryValidationError(
                    f"Agent {name!r} unknown peers: {joined}"
                )

            ordered = tuple(sorted(peers))
            peer_names[name] = ordered
            
            # Изолируем маппинг ролей для промптов
            peer_roles[name] = MappingProxyType(
                {p: roles[p] for p in ordered}
            )

        # 3. Жесткая фиксация состояния в иммутабельные структуры
        self._agents = MappingProxyType(agents)
        self._parents = MappingProxyType(parents)
        
        # Сохраняем детей как tuple для максимальной защиты
        self._children = MappingProxyType({
            n: tuple(c) for n, c in children.items()
        })
        
        self._peer_names = MappingProxyType(peer_names)
        self._peer_roles = MappingProxyType(peer_roles)
        self.entry_node = entry_node

    @property
    def agents(self) -> Mapping[str, AgentSpec]:
        """Все агенты по имени."""
        return self._agents

    def specs(self) -> tuple[AgentSpec, ...]:
        """Плоский список specs в порядке регистрации."""
        return tuple(self._agents.values())

    def get(self, name: str) -> AgentSpec:
        """Получить spec агента за O(1) или бросить исключение."""
        try:
            return self._agents[name]
        except KeyError as exc:
            raise RegistryValidationError(f"Unknown agent: {name!r}") from exc

    def peers(self, name: str) -> Mapping[str, str]:
        """Словарь peer_name -> peer_role для промпта."""
        self.get(name)
        return self._peer_roles[name]

    def peer_names(self, name: str) -> tuple[str, ...]:
        """Имена соседей для валидации transfer schema."""
        self.get(name)
        return self._peer_names[name]

    def parent_of(self, name: str) -> str | None:
        """Получить родительского агента (если это подагент)."""
        self.get(name)
        return self._parents[name]

    def children_of(self, name: str) -> tuple[str, ...]:
        """Прямые подагенты для маршрутизации вглубь."""
        self.get(name)
        return self._children.get(name, ())


agent_registry: Final[AgentRegistry] = AgentRegistry(
    DEFAULT_AGENTS, 
    entry_node=ENTRY_NODE
)
