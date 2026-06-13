"""Агенты и их prompt-протоколы."""

from app.services.swarm_agent.agents.prompts import SWARM_PROTOCOL, build_agent_prompt
from app.services.swarm_agent.agents.registry import (
    ENTRY_NODE,
    FINAL_NODE,
    INIT_NODE,
    LOOP_GUARD_NODE,
    RECOVERY_NODE,
    AgentRegistry,
    AgentSpec,
    agent_registry,
)

__all__ = [
    "ENTRY_NODE",
    "FINAL_NODE",
    "INIT_NODE",
    "LOOP_GUARD_NODE",
    "RECOVERY_NODE",
    "SWARM_PROTOCOL",
    "AgentRegistry",
    "AgentSpec",
    "agent_registry",
    "build_agent_prompt",
]
