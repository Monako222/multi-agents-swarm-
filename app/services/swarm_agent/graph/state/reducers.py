"""Редьюсеры обеспечивают бесконфликтное накопление бизнес-данных и направляют 
мутацию стейта, помогая команде изолировать логику обновления каждого канала графа. 
Для максимальной производительности копирование сложных структур реализовано на уровне Rust 
через бинарный дамп orjson, что полностью устраняет тяжелый оверхед стандартного deepcopy. 
Глубокое слияние JSON-контекста переведено на плоский итеративный обход через стек, 
защищая рантайм от падений при любой вложенности ответов моделей. Списки файлов 
и пиров гарантированно очищаются от дубликатов за линейное время $O(N)$ с 
сохранением хронологии, а журналы ошибок защищены скользящим лимитом от 
раздувания памяти. При этом рантайм поддерживает мгновенный сброс или 
перезапись каналов через сигнальные маркеры распаковки патчей."""





import orjson


from pydantic import BaseModel
from typing import Any, TypeVar

from collections.abc import (
    Iterable,
    Mapping, 
    Sequence
)


from app.services.swarm_agent.types import (
    Context, Data, Space, 
    Workspace,
    RuntimeMetrics
)





T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)
_ATOMIC_RUNTIME_TYPES = str | bytes | bytearray | Mapping | BaseModel


_ORJSON_OPTS = (
    orjson.OPT_PASSTHROUGH_DATETIME | 
    orjson.OPT_SERIALIZE_DATACLASS | 
    orjson.OPT_SORT_KEYS
)






def _dump_bytes(value: Any) -> bytes:
    """Единая точка быстрой сериализации 
    для слоя состояния. Инкапсулирует опции 
    orjson и для неизвестных типов
    """
    
    return orjson.dumps(value, option=_ORJSON_OPTS, default=str)





def _get_list(obj: Any, key: str) -> list[Any]:
    """Извлекает список без лишних 
    аллокаций памяти за O(1).
    """
    
    val = (
        getattr(obj, key, []) if isinstance(obj, BaseModel)
        else obj.get(key, []) if isinstance(obj, Mapping)
        else []
    )
    
    return val if isinstance(val, list) else []





def _get_dict(obj: Any, key: str) -> dict[str, Any]:
    """Извлекает словарь напрямую из 
    Pydantic-модели или Mapping.
    """
    
    val = (
        getattr(obj, key, {}) if isinstance(obj, BaseModel)
        else obj.get(key, {}) if isinstance(obj, Mapping)
        else {}
    )
    
    return val if isinstance(val, dict) else {}





def _unwrap(value: Any) -> tuple[bool, Any]:
    """Распознает маркер полной перезаписи 
    канала (__replace__).
    """
    
    if isinstance(value, Mapping) and "__replace__" in value:
        return True, value["__replace__"]
    return False, value





def replace_list(
    values: Sequence[T] | None = None
) -> dict[str, list[T]]:
    """Создает маркер для полной перезаписи 
    списочного канала. Заставляет редьюсер 
    забыть историю и вставить 
    свежие данные.
    """

    return {"__replace__": list(values or [])}





def replace_value(value: T) -> dict[str, T]:
    """Создает маркер для жесткой перезаписи 
    словаря  или скаляра. Отменяет 
    дефолтный deep-merge.
    """
    
    return {"__replace__": value}





def _seq(value: Any) -> tuple[Any, ...]:
    """Быстро и безопасно оборачивает любые 
    данные в кортеж. Защищает строки, 
    словари и модели от распада.
    """
    
    if value is None:
        return ()

    # Экранируем базовые и 
    # составные типы от итерации
    if isinstance(value, _ATOMIC_RUNTIME_TYPES):
        return (value,)
    
    try: 
        return tuple(value)
    except TypeError:
        return (value,)





def _stable_key(value: Any) -> Any:
    """Создает уникальный слепок (ключ) 
    для любых объектов. Позволяет безопасно 
    хэшировать словари и Pydantic-модели 
    для дедупликации.
    """
    
    try:
        # Пробуем делать
        # хэш нативно
        hash(value)
        return value
    
    except TypeError:
        # Pydantic-модели нативно 
        # выгружаем в стабильный JSON
        if isinstance(value, BaseModel):
            return value.model_dump_json(
                exclude_defaults=False
            )
            
        # Словари (Mapping) 
        # сериализуем с _dump_bytes
        if isinstance(value, Mapping):
            return _dump_bytes(value)
            
        # Безопасный фоллбэк для 
        # вложенных списков
        return repr(value)





def _dedupe(values: Iterable[T]) -> list[T]:
    """Очищает последовательность от дубликатов 
    за время O(N). Сохраняет порядок и 
    оставляет первое вхождение
    каждого элемента.
    """
    
    seen: set[Any] = set()
    out: list[T] = []
    
    for item in values:
        
        key = _stable_key(item)
        if key in seen:
            continue
            
        seen.add(key)
        out.append(item)
        
    return out





def _json_copy(value: Any) -> Any:
    """Быстрая глубокая копия 
    JSON-структур."""
    
    # Атомарные типы (str, int, bool)
    # и кортежи отдаем мгновенно как есть
    if not isinstance(value, dict | list):
        return value
        
    # Моментальный C-level дамп
    # и загрузка для создания копии
    return orjson.loads(_dump_bytes(value))





def _as_patch(value: Any) -> dict[str, Any]:
    """Нормализует сырые данные редьюсера в 
    чистый словарь (патч). Вырезает None
    значения, защищая от затирания.
    """
    
    # Быстрый возврат,
    # если обновлять нечего
    if value is None:
        return {}


    # Нативная и быстрая
    # выгрузка Pydantic-модели 
    if isinstance(value, BaseModel):
        return value.model_dump(
            exclude_unset=True, 
            exclude_none=True
        )


    # Идеальный путь для 
    # стандартных словарей
    if isinstance(value, Mapping):
        return {
            k: v for k, v 
            in value.items() 
            if v is not None
        }


    # Безопасный фоллбэк 
    # через утиную типизацию 
    return {
        k: v for k, v 
        in dict(value).items() 
        if v is not None
    }






def _merge_json(
    base: dict[str, Any], 
    patch: Mapping[str, Any]
) -> dict[str, Any]:
    """Глубокое слияние JSON-структур.
    Работает итеративно, выдерживая 
    любую глубину вложенности.
    """
    
    # Изолированная копия 
    # исходного словаря для работы
    result = _json_copy(base) if base else {}
    
    
    # Быстрый возврат,
    # если обновлять нечем
    if not patch:
        return result


    # Стек для обхода: 
    # (целевой, с_патчами)
    stack = [(result, patch)]
    
    
    while stack:
        target, source = stack.pop()
        for key, val in source.items():
            # Если с обеих сторон лежат 
            # словари — планируем слияние вглубь
            if (
                isinstance(val, Mapping) and 
                isinstance(target.get(key), dict)
            ): 
                stack.append((target[key], val))
                
            # Иначе просто переписываем 
            # узел целиком (защищая копией)
            else: target[key] = _json_copy(val)

    return result








def _merge_model(
    cls: type[M],
    cur: Any,
    inc: Any,
    *,
    unique: tuple[str, ...] = (),
) -> M:
    """Слияние Pydantic-модели с патчем. 
    Склеивает данные и уникальные списки за
    один проход без лишних аллокаций.
    """
    
    # Ранний возврат 
    # при пустом патче
    if not (patch := _as_patch(inc)):
        return (
            cur if isinstance(cur, cls) 
            else cls.model_validate(cur or {})
        )


    # Сразу готовим 
    # словарь для мутаций
    data = (
        cur.model_dump() 
        if isinstance(cur, cls) 
        else dict(cur or {})
    )


    # Проходим по безопасно 
    # скопированному патчу в один заход
    for key, val in _json_copy(patch).items():
        
        # Отрабатываем списки,
        # требующие дедупликации
        if key in unique:
            replace, values = _unwrap(val)
            
            data[key] = (
                _dedupe(_seq(values)) if replace else 
                _dedupe([*_seq(data.get(key)), *_seq(val)])
            )
            
        # Обычные поля 
        # перезаписываем
        else: data[key] = val


    # Финальная строгая 
    # валидация и сборка модели
    return cls.model_validate(data)





def r_tail(
    cur: Sequence[T] | None, 
    inc: Sequence[T] | None, 
    *, 
    limit: int = 256
) -> list[T]:
    """Ограниченный хвостовой список.
    Срезает данные за O(N), защищая память 
    от переполнения журналами.
    """
    
    replace, values = _unwrap(inc)
    if replace: 
        return list(_seq(values))[-limit:]
    return [*_seq(cur), *_seq(inc)][-limit:]





def r_unique(
    cur: Sequence[T] | None, 
    inc: Sequence[T] | None
) -> list[T]:
    """Дедуплицирующий список.
    Накапливает файлы и артефакты, 
    сохраняя исходный порядок 
    появления.
    """
    
    replace, values = _unwrap(inc)
    if replace: 
        return _dedupe(_seq(values))
    return _dedupe([*_seq(cur), *_seq(inc)])





def r_workspace(cur: Any, inc: Any) -> Workspace:
    """Слияние рабочей области. Бережно 
    сохраняет черновик, пока формируется
    финальный ответ.
    """
    
    return _merge_model(Workspace, cur, inc)





def r_context(cur: Any, inc: Any) -> Context:
    """Слияние контекста запроса. Обновляет 
    метаданные (intent, query), а список 
    тегов фильтрует от дубликатов.
    """
    
    return _merge_model(Context, cur, inc, unique=("tags",))





def r_space(cur: Any, inc: Any) -> Space:
    """Слияние общей памяти графа. Перезаписывает 
    текущие шаги (goal, step) и уникально 
    копит заметки (notes).
    """
    
    return _merge_model(Space, cur, inc, unique=("notes",))





def r_data(cur: Any, inc: Any) -> Data:
    """Объединяем бизнес-данные. Списки 
    дедуплицируем за O(N), словари 
    сливаем рекурсивно.
    """
    
    # Ранний возврат при пустом 
    # патче через моржовый оператор
    if not (patch := _as_patch(inc)):
        return (
            cur if isinstance(cur, Data) 
            else Data.model_validate(cur or {})
        )


    # Приводим текущее состояние 
    # к словарю для быстрых 
    # мутаций за O(1)
    data = (
        cur.model_dump() 
        if isinstance(cur, Data) 
        else dict(cur or {})
    )


    # Проходим строго
    # по пришедшим ключам патча
    for key, val in patch.items():
        
        # Уникальные ссылки 
        # на источники данных
        if key == "peers":
            data[key] = r_unique(
                data.get(key), val
            )
            
        # Список входящих и 
        # сгенерированных файлы
        elif key == "files":
            data[key] = r_unique(
                data.get(key), val
            )
            
        # Бизнес-факты: замена 
        # или рекурсивный deep merge
        elif key == "json_data":
            replace, values = _unwrap(val)
            data[key] = (
                _json_copy(values or {}) if replace 
                else _merge_json(data.get(key), val)
            )
            
        else:
            # Безопасный фоллбэк 
            # для остальных полей
            data[key] = _json_copy(val)


    # Собираем и строго 
    # валидируем итоговую модель
    return Data.model_validate(data)






def r_metrics(cur: Any, inc: Any) -> RuntimeMetrics:
    """Суммируем новые значения метрик с текущими.
    Инкремент работает за O(N) от размера входящего патча.
    """
    
    # Если патч пуст то сразу 
    # отдаем в ответ текущий стейт
    if not (patch := _as_patch(inc)):
        return (
            cur if isinstance(cur, RuntimeMetrics) 
            else RuntimeMetrics.model_validate(cur or {})
        )


    # Берем сырые данные
    # для инкремента
    data = (
        cur.model_dump() 
        if isinstance(cur, RuntimeMetrics)
        else dict(cur or {})
    )


    # Точечно плюсуем значения
    for k, v in patch.items():
        if k in RuntimeMetrics.model_fields:
            data[k] = int(data.get(k, 0)) + int(v or 0)


    # Выдаем собранную валидную модель
    return RuntimeMetrics.model_validate(data)
