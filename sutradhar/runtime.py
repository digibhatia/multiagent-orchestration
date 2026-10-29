"""Thin helpers for RUNNING the graph - shared by the UI, the demo script and the comparison script.

Each question is one THREAD (thread_id). The checkpointer stores that thread's state, so:
  start()  -> runs until the graph ends OR pauses at human_review
  resume() -> continues the paused thread with the human's decision
"""
import uuid

from langgraph.types import Command

from . import config


def new_thread_id() -> str:
    return "q-" + uuid.uuid4().hex[:8]


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _interrupt_payload(graph, result: dict, thread_id: str):
    """Return the payload shown to the human, or None if the run is not paused."""
    intr = result.get("__interrupt__") if isinstance(result, dict) else None
    if intr:
        return getattr(intr[0], "value", intr[0])
    snap = graph.get_state(_cfg(thread_id))                    # fallback for older LangGraph versions
    if snap.next:
        for task in snap.tasks:
            if getattr(task, "interrupts", None):
                return task.interrupts[0].value
    return None


def _outcome(graph, result: dict, thread_id: str) -> dict:
    payload = _interrupt_payload(graph, result, thread_id)
    state = graph.get_state(_cfg(thread_id)).values            # full, merged state (also valid while paused)
    if payload is not None:
        return {"status": "waiting", "thread_id": thread_id, "interrupt": payload, "state": state}
    return {"status": "done", "thread_id": thread_id, "state": state}


def start(graph, question: str, route_mode: str = config.DEFAULT_ROUTE_MODE,
          require_approval: bool = config.REQUIRE_APPROVAL, thread_id: str = None) -> dict:
    thread_id = thread_id or new_thread_id()
    result = graph.invoke({"question": question, "route_mode": route_mode, "require_approval": require_approval,
                           "steps": [], "writer_attempts": 0}, config=_cfg(thread_id))
    return _outcome(graph, result, thread_id)


def resume(graph, thread_id: str, approve: bool, note: str = "") -> dict:
    result = graph.invoke(Command(resume={"approve": approve, "note": note}), config=_cfg(thread_id))
    return _outcome(graph, result, thread_id)


def totals(steps: list) -> dict:
    return {"llm_calls": sum(s.get("llm_calls", 0) for s in steps),
            "input_tokens": sum(s.get("input_tokens", 0) for s in steps),
            "output_tokens": sum(s.get("output_tokens", 0) for s in steps),
            "seconds": round(sum(s.get("seconds", 0) for s in steps), 2)}
