"""Фасад пакета swarm-agent внутри приложения."""

from __future__ import annotations

import sys

from app.services.swarm_agent.config import Settings, get_settings

# Совместимость со старым standalone-именем пакета без изменения структуры папок.
sys.modules.setdefault("swarm_agent", sys.modules[__name__])

__all__ = ["Settings", "get_settings"]
