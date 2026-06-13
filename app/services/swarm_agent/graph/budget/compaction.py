"""Сжатие message history без LLM-вызова.

Главная цель модуля — не просто обрезать хвост, а сохранить абсолютную 
валидность истории для tool-calling моделей: AIMessage с `tool_calls` 
строго должен сопровождаться соответствующими ToolMessage. 
Иначе провайдер отклонит следующий запрос.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.services.swarm_agent.text import dedupe_lines, dumps_compact, tail_text, truncate_text


@dataclass(frozen=True, slots=True)
class MessageBlock:
    """Неразрывный блок истории: обычное сообщение или AI+ToolMessages."""

    messages: tuple[Any, ...]
    valid_for_prompt: bool = True
    orphan_count: int = 0


@dataclass(frozen=True, slots=True)
class MessageCompaction:
    """Итоговый результат сжатия истории сообщений."""

    kept: list[Any]
    remove_ids: list[str]
    episodic_memory: str
    evicted_count: int = 0
    dropped_orphans: int = 0
    unremovable_count: int = 0


def _message_role(msg: Any) -> str:
    """O(1) нормализация роли без привязки к классам LangChain."""
    
    return str(
        getattr(msg, "type", None)
        or getattr(msg, "role", None)
        or msg.__class__.__name__
    ).lower()


def _message_content_text(content: Any) -> str:
    """Быстрое извлечение текста из строк, словарей и мультимодальных блоков."""
    
    if not content:
        return ""
        
    if isinstance(content, str):
        return content
        
    if isinstance(content, dict):
        return dumps_compact(content)
        
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
            elif "text" in item:
                parts.append(str(item["text"]))
            elif item_type := item.get("type"):
                parts.append(f"<{item_type}>")
            else:
                parts.append(truncate_text(dumps_compact(item), 500))
        return " ".join(parts)
        
    return str(content)


def _tool_call_ids(msg: Any) -> list[str]:
    """Сбор ID всех вызовов инструментов за O(N)."""
    
    if not (calls := getattr(msg, "tool_calls", None)):
        return []
        
    res: list[str] = []
    for c in calls:
        cid = c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
        if cid:
            res.append(str(cid))
            
    return res


def _tool_message_call_id(msg: Any) -> str | None:
    """Извлечение ID из ответа инструмента (ToolMessage)."""
    
    if cid := getattr(msg, "tool_call_id", None):
        return str(cid)
    return None


def _is_tool_message(msg: Any) -> bool:
    """Моментальная проверка на инструмент-результат."""
    
    return _message_role(msg) in {"tool", "toolmessage"}


def _has_tool_calls(msg: Any) -> bool:
    """Проверка наличия запросов к инструментам внутри AI-ответа."""
    
    return bool(getattr(msg, "tool_calls", None))


def _message_id(msg: Any) -> str | None:
    """Безопасное извлечение ID сообщения для его удаления из State."""
    
    if mid := getattr(msg, "id", None):
        return str(mid)
    return None


def _message_blocks(messages: Sequence[Any]) -> list[MessageBlock]:
    """Разбить историю на prompt-safe блоки.

    ToolMessage без AIMessage считается orphan-блоком и исключается. 
    AIMessage с неполным набором ответов переносится в episodic_memory.
    """
    
    blocks: list[MessageBlock] = []
    total = len(messages)
    i = 0

    while i < total:
        msg = messages[i]

        if _is_tool_message(msg):
            blocks.append(
                MessageBlock((msg,), valid_for_prompt=False, orphan_count=1)
            )
            i += 1
            continue

        if not _has_tool_calls(msg):
            blocks.append(
                MessageBlock((msg,), valid_for_prompt=True)
            )
            i += 1
            continue

        expected = set(_tool_call_ids(msg))
        tool_msgs: list[Any] = []
        actual_ids: list[str] = []
        
        j = i + 1
        while j < total and _is_tool_message(messages[j]):
            t_msg = messages[j]
            tool_msgs.append(t_msg)
            
            if cid := _tool_message_call_id(t_msg):
                actual_ids.append(cid)
            j += 1

        # Требуем строгого совпадения: без пропусков и без лишних ответов
        actual = set(actual_ids)
        is_valid = (
            bool(expected) 
            and actual == expected 
            and len(actual_ids) == len(expected)
        )
        
        block_msgs = (msg, *tool_msgs)
        blocks.append(
            MessageBlock(
                messages=block_msgs,
                valid_for_prompt=is_valid,
                orphan_count=0 if is_valid else len(block_msgs),
            )
        )
        i = j

    return blocks


def _select_kept_blocks(blocks: Sequence[MessageBlock], keep_last: int) -> set[int]:
    """Выбрать последние валидные блоки, не разрывая связи вызовов."""
    
    if keep_last <= 0:
        return set()

    kept: set[int] = set()
    used = 0

    # Обратный цикл для сохранения самых свежих данных
    for idx in range(len(blocks) - 1, -1, -1):
        block = blocks[idx]
        
        if not block.valid_for_prompt:
            continue

        size = len(block.messages)
        
        if not kept:
            # Гарантируем хотя бы один блок: лучше превысить лимит, 
            # чем отправить LLM абсолютно пустой контекст
            kept.add(idx)
            used += size
            continue

        if used + size > keep_last:
            break

        kept.add(idx)
        used += size

    return kept


def render_message_for_memory(msg: Any, *, max_chars: int = 700) -> str:
    """Сжать одно сообщение до строки для помещения в episodic memory."""
    
    role = _message_role(msg)
    content = truncate_text(_message_content_text(getattr(msg, "content", "")), max_chars)
    
    if calls := getattr(msg, "tool_calls", None):
        # Быстрая генерация списка имен инструментов
        names = [
            str(c.get("name", "unknown") if isinstance(c, dict) else getattr(c, "name", "unknown"))
            for c in calls
        ]
        content = f"{content}\nTool calls: {', '.join(names)}".strip()
        
    return f"{role}: {content}".strip()


def compact_messages(
    messages: Sequence[Any],
    *,
    keep_last: int,
    previous_summary: str = "",
    max_summary_chars: int,
) -> MessageCompaction:
    """Оставить prompt-safe хвост сообщений, старое перенести в память."""
    
    blocks = _message_blocks(messages)
    kept_indexes = _select_kept_blocks(blocks, keep_last)

    kept: list[Any] = []
    evicted: list[Any] = []
    dropped_orphans = 0

    for idx, block in enumerate(blocks):
        if idx in kept_indexes:
            kept.extend(block.messages)
        else:
            evicted.extend(block.messages)
            dropped_orphans += block.orphan_count

    if not evicted:
        return MessageCompaction(
            kept=kept,
            remove_ids=[],
            episodic_memory=previous_summary,
        )

    # Моментальная сборка новой памяти через list comprehension
    rendered = "\n".join([render_message_for_memory(m) for m in evicted])
    combined = f"{previous_summary}\n{rendered}".strip() if previous_summary else rendered
    summary = tail_text(dedupe_lines(combined), max_summary_chars)

    # Словарь dict используется как упорядоченный set (стандарт Python 3.7+)
    # Это позволяет собрать уникальные ID быстрее, чем старый вариант
    remove_ids: dict[str, None] = {}
    unremovable = 0
    
    for m in evicted:
        if msg_id := _message_id(m):
            remove_ids[msg_id] = None
        else:
            unremovable += 1

    return MessageCompaction(
        kept=kept,
        remove_ids=list(remove_ids),
        episodic_memory=summary,
        evicted_count=len(evicted),
        dropped_orphans=dropped_orphans,
        unremovable_count=unremovable,
    )
