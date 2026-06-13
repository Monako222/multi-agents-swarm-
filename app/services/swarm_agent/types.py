"""Единый файл доменных типов и лёгких Pydantic-моделей.
Здесь собраны контракты данных, которые используются несколькими слоями:
state, agents, tools, llm и service. В файле нет сетевых клиентов, LangGraph
узлов и тяжёлой runtime-логики, поэтому его можно импортировать в тестах и
утилитах без побочных эффектов.
"""



from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any
from collections.abc import Mapping
from datetime import datetime, timezone
from pydantic import (
    StringConstraints, 
    field_validator,
    ConfigDict, 
    SecretStr,
    BaseModel, 
    Field, 
)



# Поддержка языковых кодов: от "en" 
# до сложных конструкций вроде "zh-Hant"
LANG_RX = r"(?i)^[a-z]{2,8}(?:-[a-z0-9]{2,8})?$"


# Строгие имена для системных узлов 
# (нижний регистр, цифры, подчеркивания)
AGENT_RX = r"^[a-z0-9_]+$"




type AtomicData = (
    str 
    | bytes 
    | bytearray 
    | Mapping 
    | BaseModel
)



type JsonAtom = (
    str 
    | int 
    | float 
    | bool 
    | None
)



type JsonValue = (
    JsonAtom 
    | list[JsonValue] 
    | dict[str, JsonValue]
)



type JsonObject = dict[str, JsonValue]
type RawSnapshot = (Mapping[str, Any] | SwarmSnapshot)



type NonEmptyStr = Annotated[
    str, StringConstraints(
        strip_whitespace=True,
        min_length=1,
    )
]



type ShortText = Annotated[
    str, StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=300,
    )
]



type LongText = Annotated[
    str, StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=4_000,
    )
]



type FinalText = Annotated[
    str, StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=40_000,
    )
]



type Identifier = Annotated[
    str, StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=512,
    )
]



type TagText = Annotated[
    str, StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
    )
]



type LangCode = Annotated[
    str, StringConstraints(
        strip_whitespace=True,
        pattern=LANG_RX,
        min_length=2,
        max_length=16
    )
]



type AgentName = Annotated[
    str, StringConstraints(
        strip_whitespace=True,
        pattern=AGENT_RX,
        min_length=1
    )
]



type ToolName = Annotated[
    str, StringConstraints(
        strip_whitespace=True,
        pattern=AGENT_RX,
        min_length=1
    )
]






class FileKind(StrEnum):
    """Тип файла, видимый 
    агентам через state."""

    DOC = "doc"
    PHOTO = "photo"
    VIDEO = "video"
    VOICE = "voice"
    AUDIO = "audio"
    OTHER = "other"



class OutputMode(StrEnum):
    """Формат результата, который 
    агент может принять от 
    peer/sub-agent."""

    TEXT = "text"
    FILE = "file"
    JSON = "json"




class ErrorSeverity(StrEnum):
    """Уровень технической 
    записи в журнале state."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"




class TransferProtocol(StrEnum):
    """Протокол передачи 
    управления между 
    агентами."""

    LOCAL = "local"
    A2A = "a2a"





class ProviderName(StrEnum):
    """LLM-провайдеры, 
    поддержанные текущей 
    фабрикой."""

    OPENROUTER = "openrouter"






class Modality(StrEnum):
    """Модальности данных 
    для подбора модели."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCS = "document"






class ToolCategory(StrEnum):
    """Категории инструментов 
    для каталога."""

    SYSTEM = "system"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    DOCS = "docs"
    WEB = "web"






class MutableStateModel(BaseModel):
    """База для state-каналов, 
    которые изменяются редьюсерами."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
        validate_by_name=True,
        validate_by_alias=True,
    )





class FrozenStateModel(BaseModel):
    """База для неизменяемых 
    артефактов, журналов и specs."""

    model_config = ConfigDict(
        validate_by_alias=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
        validate_by_name=True,
        extra="forbid",
        frozen=True,
    )








class Workspace(MutableStateModel):
    """Канал пользовательского ответа:
    черновик и финальная строка."""

    # Промежуточный вариант 
    # текста, формируемый агентами
    draft_answer: str = ""
    
    # Окончательный выверенный
    # ответ, готовый к мгновенной 
    # демонстрации пользователю
    final_answer: str = ""




class File(FrozenStateModel):
    """Входной или выходной файл, 
    видимый агентам через 
    runtime context."""

    # Уникальный 
    # идентификатор
    id: Identifier

    # Категория контента 
    # для выбора обработки 
    kind: FileKind = FileKind.DOC
    
    # Прямая ссылка 
    # или путь к файлу
    uri: str = ""
    
    # Краткое текстовое 
    # сути файла для LLM
    desc: str = ""
    
    #-------------------------------
    # TODO: что если будет ????дохулион????
    #-------------------------------
    # Извлеченный 
    # текстовый контент 
    content: str = ""








class Context(MutableStateModel):
    """Нормализованная семантика 
    текущего пользовательского 
    запроса."""

    # Исходный сырой 
    # текст сообщения
    query: str = ""
    
    # Выявленная
    # цель обращения
    intent: str = ""
    
    # Главная тема, 
    # предмет или сущность
    subject: str = ""
    
    # Распознанный 
    # код языка общения 
    lang: str = ""
    
    # Список извлеченных 
    # тегов, дедуплицируемый редьюсером
    tags: list[TagText] = Field(default_factory=list)







class Space(MutableStateModel):
    """Общая память роя 
    для текущей задачи и 
    сжатой истории."""

    # Глобальная цель 
    # или верхнеуровневое ТЗ
    goal: str = ""
    
    # Описание текущего 
    # шага или подзадачи 
    step: str = ""
    
    # Краткая выжимка 
    # промежуточных итогов
    brief: str = ""

    # Долгосрочный контекст, 
    # который выдерживает компрессию
    episodic_memory: str = ""
    
    # Скользящий список коротких 
    # заметок, дедуплицируемый редьюсером
    notes: list[ShortText] = Field(default_factory=list)







class LinkRecord(FrozenStateModel):
    """Ссылка или верифицированный 
    внешний источник, сохранённый 
    агентом при поиске."""

    # Уникальный 
    # идентификатор 
    id: str = ""
    
    # Прямой URL-адрес 
    # внешнего веб-ресурса
    url: str = ""
    
    # Заголовок веб-стр
    # статьи или другого
    title: str = ""
    
    # Текстовый фрагмент 
    # (сниппет) для передачи
    snippet: str = ""
    
    # Имя поискового инструмента 
    # или движка, который вернул эту ссылку
    source: str = ""








class ErrorRecord(FrozenStateModel):
    """Структурированная техническая запись
    о техническом сбое или проблеме.
    """

    # Описание возникшей
    # ошибки или трейсбэк
    message: NonEmptyStr
    
    # Идентификатор компонента 
    # или имя агента, где ошибка
    source: NonEmptyStr = "swarm"
    
    # Уровень критичности проблемы
    # (от простого инфо до фатального)
    severity: ErrorSeverity = ErrorSeverity.ERROR
    
    # Точная временная
    # метка фиксации ошибки 
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )






class Data(MutableStateModel):
    """Структурированные бизнес-данные, 
    накопленные агентами в ходе выполнения задач.
    Все поля обновляются через специализированные 
    изолированные редьюсеры.
    """

    # Список внешних 
    # источников, ссылок и др
    peers: list[LinkRecord] = Field(default_factory=list)
    
    # Свободный JSON 
    # контекст для хранения фактов 
    json_data: JsonObject = Field(default_factory=dict)
    
    # Файлы загруженные или 
    # сгенерированных роем файлов
    files: list[File] = Field(default_factory=list)








class PendingTransfer(FrozenStateModel):
    """Описание текущей передачи управления между агентами.
    Действует как неизменяемый контракт (трансфер) 
    при переключении контекста выполнения.
    """

    # Идентификатор целевого агента,
    # которому дается задача
    target_agent: Identifier
    
    # Идентификатор инициатора 
    # передачи контроля
    requested_by: Identifier
    
    # Текстовое описание или 
    # ТЗ для принимающего агента
    task_description: str = ""
    
    # Внешний сетевой эндпоинт для 
    # удаленной передачи по A2A
    endpoint_uri: str | None = None
    
    # Ссылка на паспорт или
    # манифест характеристик агента
    agent_card_uri: str | None = None
    
    # Протокол маршрутизации
    # (локальный или межплатформенный)
    protocol: TransferProtocol = TransferProtocol.LOCAL
    
    # Список ожидаемых форматов ответа 
    # (текст, файл, структурированный JSON)
    accepted_output_modes: tuple[OutputMode, ...] = (
        OutputMode.TEXT,
    )
    
    # Точная временная 
    # метка создания трансфера
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )




class RuntimeMetrics(MutableStateModel):
    """Накопительные метрики выполнения текущего thread/run."""

    # Общее количество
    # выполненных запросов
    llm_calls: int = 0
    
    # Количество
    # инструментов
    tool_calls: int = 0
    
    # Число retries
    # при сбоях в API
    retries: int = 0
    
    # Количество старых сообщений, 
    # упакованных в эпизодическую память
    compacted_messages: int = 0






class SwarmSnapshot(MutableStateModel):
    """Типизированный снимок 
    полного состояния графа."""

    # Текущее число циклов 
    # активного агента
    loops: int = 0
    
    # Общее количество
    # шагов во всем графе
    total_steps: int = 0
    
    # Имя или идентификатор 
    # агента, который активен сейчас
    active_node: str | None = None
    
    # Накопленные различные
    # бизнес-данные и ссылки для работы
    data: Data = Field(default_factory=Data)
    
    # Общая разделяемая 
    # память роя и скользящие заметки
    space: Space = Field(default_factory=Space)
    
    # Нормализованные метаданные 
    # исходного запроса пользователя
    context: Context = Field(default_factory=Context)
    
    # Рабочая область с 
    # черновиками и готовым ответом
    workspace: Workspace = Field(default_factory=Workspace)
    
    # Ожидающий выполнения 
    # контракт на передачу контроля
    pending_transfer: PendingTransfer | None = None
    
    # Полная история сообщений 
    # и диалогов внутри роя
    messages: list[Any] = Field(default_factory=list)
    
    # Лог переходов между 
    # узлами для трассировки пути
    history: list[str] = Field(default_factory=list)
    
    # Список исходных артефактов и
    # файлов, переданных на обработку
    in_files: list[File] = Field(default_factory=list)
    
    # Список нв выход артефактов и 
    # файлов, созданных роем на выдачу
    out_files: list[File] = Field(default_factory=list)
    
    # Журнал зафиксированных 
    # технических сбоев и ошибок
    errors: list[ErrorRecord] = Field(default_factory=list)
    
    # Суммарные рантайм
    # метрики выполнения текущего потока
    metrics: RuntimeMetrics = Field(default_factory=RuntimeMetrics)
    
    # Сигнальный флаг 
    # завершения работы роя
    is_final: bool = False







#-------------------------------
# TODO: НЕПОНЯТНО ЗАЧЕМ - УДАЛИТЬ?
# Хотя воможно нужная вещь
#-------------------------------

class SwarmResult(BaseModel):
    """Стабильный DTO ответа фасада для 
    внешнего application-кода. Инкапсулирует 
    итоговый текст, метаданные потока и
    полный слепок состояния графа.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    # Уникальный идентификатор 
    # текущей сессии или диалога
    thread_id: str

    # Текст ответа 
    # для пользователя
    final_answer: str
    
    # Валидированный снимок 
    # полного состояния агентов
    snapshot: SwarmSnapshot
    
    # Сырой словарь состояния 
    # из LangGraph для отладки и логов
    raw_state: dict[str, Any] = Field(default_factory=dict)
    
    # Флаг успешности: True, 
    # что отработал без сбоев
    ok: bool = True


    @property
    def errors(self) -> list[ErrorRecord]:
        """Предоставляет прямой доступ к
        журналу технических ошибок."""
        
        return self.snapshot.errors







class ProviderConfig(FrozenStateModel):
    """Настройки подключения 
    к OpenAI-compatible 
    провайдеру."""

    # Название из 
    # поддерживаемых
    name: ProviderName
    
    # Базовый юрл
    # для подлючения
    base_url: NonEmptyStr
    
    # Флаг использовать  
    # ли трафик через прокси
    use_proxy: bool = False
    
    # Секретный токен 
    # доступа из настроек
    api_key: SecretStr | None = None
    
    # Максимальное число
    # ретраев клиента при сбоях
    max_retries: int = Field(default=3, ge=0, le=10)
    
    # Общий лимит времени на 
    # чтение и выдачу ответа от модели
    timeout_s: float = Field(default=120.0, ge=1.0, le=600.0)
    
    # Лимит времени на создание 
    # TCP-соединения с провайдером
    connect_timeout_s: float = Field(default=15.0, ge=1.0, le=120.0)
    
    # Время ожидания выделения 
    # свободного слота внутри HTTP-пула
    pool_timeout_s: float = Field(default=15.0, ge=1.0, le=120.0)
    
    # Дополнительные HTTP заголовки 
    # (атрибуция, кастомные ключи и другое)
    headers: Mapping[str, str] = Field(default_factory=dict)








class ModelProfile(FrozenStateModel):
    """Технический профиль 
    LLM-модели."""

    # Уникальный 
    # ключ для кэша 
    alias: NonEmptyStr
    
    # Системное имя 
    # модели для API
    model_id: NonEmptyStr
    
    # Название 
    # провайдера модели  
    provider: ProviderName
    
    # Максимальный размер
    # входного окна в токенах
    context_window: int = Field(gt=0)
    
    # Метки для роутинга и 
    # фильтрации (fast, cheap, reasoning)
    tags: frozenset[str] = Field(default_factory=frozenset)
    
    # Креативность ответов 
    # (0.0 - строго, 2.0 - максимум)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    

    # Поддерживаемые форматы 
    # данных (текст, картинки, документы)
    modalities: frozenset[Modality] = Field(
        default_factory=lambda: frozenset({
            Modality.TEXT
        })
    )
    
    # Запасные алиасы для
    # фолбека основной модели
    fallbacks: tuple[str, ...] = ()
    
    # Лимит генерации
    # (None оставляет дефолт провайдера)
    max_output_tokens: int | None = Field(default=None, gt=0)
    
    # Дополнительные специфичные 
    # параметры (top_p, seed и т.д.)
    extra_body: Mapping[str, Any] = Field(default_factory=dict)


    def supports(self, modality: Modality) -> bool:
        """Проверка поддержки формата модели."""
        return modality in self.modalities






    
    

class AgentSpec(FrozenStateModel):
    """Паспорт локального агента или подагента.
    ``children`` позволяет описывать дерево подагентов. 
    Реестр сам добавляет связь parent <-> child в 
    peer-routing,  поэтому подагент может вернуть
    ответ родителю обычным ``transfer``."""
    
    # Уникальное имя 
    # в нашем реестре
    name: AgentName
    
    # Промпт/описание роли, 
    # определяющее поведение
    role: NonEmptyStr
    
    # Специфичные инструкции 
    # или строгие запреты для роли
    rules: NonEmptyStr | None = None
    
    # Список обязанностей 
    # или задачей для агента 
    tasks: tuple[NonEmptyStr, ...]
    
    # Набор системных 
    # инструментов и API
    tools: tuple[ToolName, ...] = ()
    
    # Коллеги агенты, которым 
    # разрешено передавать handoffs
    peers: tuple[AgentName, ...] = ()
    
    # Вложенные подагенты 
    # для выстраивания иерархии роя
    children: tuple["AgentSpec", ...] = ()
    
    # Точечное переопределение LLM 
    # (например, если узлу нужен o1-preview)
    model_alias: NonEmptyStr | None = None
    
    # Жесткий лимит шагов рассуждения 
    # для защиты от бесконечных циклов
    max_local_loops: int | None = Field(default=None, ge=1, le=100)
    

    @field_validator("tasks")
    @classmethod
    def _tasks_not_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Очищаем дубликаты за и гарантируем 
        наличие хотя бы одной задачи."""
        
        cleaned = tuple(dict.fromkeys(value))
        if not cleaned:
            raise ValueError(
                "Agent must have "
                "at least one task."
            )
        return cleaned


    @field_validator("peers", "tools")
    @classmethod
    def _dedupe_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Удаляем дубликаты из промптов и схем, 
        сохраняя исходный порядок."""
        
        return tuple(dict.fromkeys(value))



# Обязательная сборка модели
# для разрешения рекурсивной ссылки
AgentSpec.model_rebuild()

