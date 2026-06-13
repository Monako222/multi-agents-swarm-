"""Real first-run check for the local swarm-agent graph.

Run:
    uv run python test_swarm_first_run.py "Привет друг, как дела твои?"

Required:
    OPENROUTER_API_KEY or SWARM_OPENROUTER_API_KEY

This is an end-to-end graph call. It sends a HumanMessage into LangGraph, lets
the configured agent call the real LLM provider, and prints the final answer
from graph state.
"""

from __future__ import annotations

import asyncio
import sys
from uuid import uuid4

from langchain_core.messages import HumanMessage

from app.services.swarm_agent.config import get_settings
from app.services.swarm_agent.graph.builder import build_swarm_graph
from app.services.swarm_agent.graph.state import SwarmState, get_answer, to_snapshot
from app.services.swarm_agent.observability import get_tracing_manager


DEFAULT_QUERY = "Привет друг, как дела твои? Ответь коротко, одним предложением. Найди мне новые новости за последние пару дней иил недель последниеновые ноовсти в интернете по новостям открытиям в атомной энергетике, что нового и другое,  также важно покажи откуда взяты данные с каких источников информации!"


def query_from_argv() -> str:
    query = " ".join(sys.argv[1:]).strip()
    return query or DEFAULT_QUERY


def require_api_key() -> None:
    settings = get_settings()
    if settings.OPENROUTER_API_KEY is None:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. Set OPENROUTER_API_KEY "
            "or SWARM_OPENROUTER_API_KEY and run this script again."
        )


async def run_graph(query: str) -> str:
    settings = get_settings()
    graph = build_swarm_graph(settings=settings)
    thread_id = f"first-run-{uuid4().hex[:12]}"
    tracing = get_tracing_manager()

    state: SwarmState = {
        "messages": [HumanMessage(content=query)],
        "in_files": [],
    }
    config = tracing.runnable_config(
        base={"recursion_limit": settings.max_total_steps + 16},
        thread_id=thread_id,
        user_id="local-first-run",
        session_id=thread_id,
        tags=("first-run", "e2e"),
        metadata={"entry": "test_swarm_first_run"},
    )

    print(f"QUERY: {query}")
    print(f"THREAD: {thread_id}")
    print("RUN: graph.ainvoke -> real agent/model call")

    try:
        raw_state = await graph.ainvoke(state, config=config)
    finally:
        await tracing.close()
    snapshot = to_snapshot(raw_state)
    answer = get_answer(snapshot)

    print("\nANSWER:")
    print(answer)
    print("\nSTATE:")
    print(f"active_node={snapshot.active_node!r}")
    print(f"is_final={snapshot.is_final}")
    print(f"messages={len(snapshot.messages)}")
    print(f"llm_calls={snapshot.metrics.llm_calls}")
    print(f"tool_calls={snapshot.metrics.tool_calls}")
    print(f"errors={len(snapshot.errors)}")

    if not answer.strip():
        raise AssertionError("Graph completed without final answer")
    if snapshot.metrics.llm_calls < 1:
        raise AssertionError("Graph did not call the LLM")

    return answer


def main() -> None:
    require_api_key()
    asyncio.run(run_graph(query_from_argv()))


if __name__ == "__main__":
    main()
