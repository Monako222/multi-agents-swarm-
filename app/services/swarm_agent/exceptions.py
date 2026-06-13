"""Доменные исключения сервиса роя.

Исключения не смешиваются с пользовательскими ответами. Внешний фасад пишет
техническую причину в snapshot/loguru, а пользователю отдаёт спокойный текст
без внутренних деталей реализации.
"""

from __future__ import annotations


class SwarmError(RuntimeError):
    """Базовая recoverable-ошибка сервиса."""


class ConfigurationError(SwarmError):
    """Конфигурация неполная, небезопасная или противоречивая."""


class MissingApiKeyError(ConfigurationError):
    """LLM-провайдер запрошен без настроенного API-ключа."""


class RegistryValidationError(ConfigurationError):
    """Топология агентов или моделей не прошла валидацию."""


class ContextBudgetError(SwarmError):
    """Ошибка подготовки или сжатия контекста."""

class ToolExecutionError(SwarmError):
    """Инструмент завершился с ошибкой, от которой граф должен восстановиться."""


class GraphExecutionError(SwarmError):
    """Граф завершился аварийно вне штатных узлов восстановления."""
