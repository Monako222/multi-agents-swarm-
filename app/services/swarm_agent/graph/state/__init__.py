"""Слой состояния state — это единый источник правды для всего роя, 
координирующий работу графа без участия скрытых супервизоров. Все каналы 
построены на строгих моделях Pydantic с ограничением extra="forbid", полностью 
защищающим рантайм от невалидных данных. Накопительные данные, файлы и 
логи обновляются через изолированные редьюсеры, а также управляющие 
триггеры перезаписываются последним апдейтом. Для максимальной 
производительности под нагрузкой в Python 3.12 внедрены 
фильтрация по схеме и извлечение финального ответа за 
$O(1)$ в обход тяжелого парсинга."""





from collections.abc import Mapping, Sequence
from typing import Annotated, Any, NotRequired, TypedDict
from langchain_core.messages import convert_to_messages
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


from app.services.swarm_agent.types import (
    Context, Data, Space, File,
    ErrorRecord, PendingTransfer,
    RuntimeMetrics, SwarmSnapshot,
    RawSnapshot, Workspace
)


from .reducers import (
    replace_list,
    replace_value,
    r_context, r_data, r_metrics,
    r_space, r_tail, r_unique,
    r_workspace,
)







class SwarmState(TypedDict):
    """Каналы общего состояния роя. 
    Каналы с ``Annotated`` имеют редьюсеры. 
    Остальные намеренно перезаписываются последним
    update: ``active_node``, ``pending_transfer``, 
    ``is_final``.
    """

    # Текущее количество 
    # итераций работы роя
    loops: NotRequired[int]
    
    # Общее количество 
    # выполненных шагов в графе
    total_steps: NotRequired[int]
    
    # Имя узла или агента,
    # который активен прямо сейчас
    active_node: NotRequired[str | None]
    
    # Набор структурированных 
    # бизнес-данных, которые собирает рой
    data: NotRequired[Annotated[Data, r_data]]
    
    # Разделяемое пространство 
    # или общая среда выполнения агентов
    space: NotRequired[Annotated[Space, r_space]]
    
    # Глобальный контекст задачи,
    # промпты и системные настройки
    context: NotRequired[Annotated[Context, r_context]]
    
    # Полная история 
    # сообщений внутри роя
    messages: Annotated[list[AnyMessage], add_messages]
    
    # Ожидающая выполнения 
    # передача контроля агенту
    pending_transfer: NotRequired[PendingTransfer | None]
    
    # Рабочая область со всеми 
    # временными артефактами и файлами
    workspace: NotRequired[Annotated[Workspace, r_workspace]]

    # Лог последних ошибок, 
    # зафиксированных во время работы
    errors: NotRequired[Annotated[list[ErrorRecord], r_tail]]
    
    # История переходов по 
    # узлам графа для трекинга пути
    history: NotRequired[Annotated[list[str], r_tail]]
    
    # Уникальный список входных
    # файлов, переданных на обработку
    in_files: NotRequired[Annotated[list[File], r_unique]]
    
    # Уникальный список выходных
    # файлов, сгенерированных агентами
    out_files: NotRequired[Annotated[list[File], r_unique]]
    
    # Технические метрики 
    # рантайма (время, токены, затраты)
    metrics: NotRequired[Annotated[RuntimeMetrics, r_metrics]]
    
    # Флаг завершения работы
    is_final: NotRequired[bool]
    
    
    
    
    
    
    
def to_snapshot(
    raw: Any
) -> SwarmSnapshot:
    """Формируем снимок состояния роя.
    Фильтруем данные, строго валидируем.
    """
    
    # Моментальный 
    # возврат готовой модели
    if isinstance(raw, SwarmSnapshot):
        return raw
        
        
    # Защита от ошибочных структур
    if not isinstance(raw, Mapping):
        raise ValueError(
            "Ошибочный формат: "
            f"{type(raw).__name__}"
        )


    # Оставляем строго 
    # известные поля у модели
    fields = SwarmSnapshot.model_fields
    data = {k: raw[k] for k in fields if k in raw}
    
    
    msgs = raw.get("messages")
    if ( # Приводим к стандарту LangChain
        msgs and isinstance(msgs, Sequence) 
        and not isinstance(msgs, str | bytes)
    ):
        data["messages"] = convert_to_messages(msgs)
        
        
    # Возвращаем собранную 
    # Pydantic-модель снимка состояния
    return SwarmSnapshot.model_validate(data)







def get_answer(
    raw: RawSnapshot
) -> str:
    """Безопасно извлекает финальный 
    текст из стейта. Работает за O(1) 
    с объектами и сырыми словарями 
    без аллокаций.
    """
    
    fallback = (
        "Не получилось сформировать полный ответ. "
        "Попробуйте, пожалуйста, чуть позже.."
    )


    # 1. На вход получена 
    # готовая модель данных
    if isinstance(raw, SwarmSnapshot):
        work = raw.workspace
        return (
            work.final_answer 
            or work.draft_answer 
            or fallback
        )


    # 2. На вход получен 
    # сырой словарь данных
    if isinstance(raw, Mapping):
        space = raw.get("workspace") or {}
        
        
        # Если у нас получен
        # воркспейс как словарь
        if isinstance(space, Mapping):
            return (
                space.get("final_answer") 
                or space.get("draft_answer") 
                or fallback
            )
            
            
        # Если у нас получен
        # воркспейс как объект
        return (
            getattr(space, "final_answer", "") 
            or getattr(space, "draft_answer", "") 
            or fallback
        )


    return fallback




__all__ = [
    "SwarmState", 
    "Context",
    "Data",
    "Space",
    "File",
    "ErrorRecord",
    "PendingTransfer",
    "RuntimeMetrics",
    "SwarmSnapshot",
    "Workspace",
    "replace_list",
    "replace_value",
    "to_snapshot",
    "get_answer",
]
