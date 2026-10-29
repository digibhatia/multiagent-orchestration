"""THE ORCHESTRATOR: a LangGraph state machine that wires the agents together.

LangGraph vocabulary (see the guide):
  STATE  - one shared dictionary (class State). Every node reads it and returns the keys it wants to change.
  NODE   - a Python function(state) -> partial state update. A node may call an LLM agent, or just run code.
  EDGE   - "after A, go to B".  CONDITIONAL EDGE - "after A, call this function; it returns the next node's name".
  CHECKPOINTER - saves the state after every node, which is what lets a run PAUSE for a human and RESUME later.

              ┌──────────────► direct_rag ─────────────────────────────────────────────► END
   START ► router
              └─► retrieval ─┬─(nothing relevant)─► refuse ───────────────────────────────► END
                             └─► analyst ─┬─(needs review)─► human_review ─┬─(approved)─┐
                                          └────────────────────────────────┘            ▼
                                                                                      writer ◄──┐
                                                                                        ▼       │ retry (feedback)
                                                                                     validate ──┘
                                                                                  ok │      │ fail
                                                                               finalize   fallback ► END
"""
import operator
import time
from typing import Annotated, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from . import agents, config, logic


class State(TypedDict, total=False):
    # inputs
    question: str
    route_mode: str            # auto | always_direct | always_pipeline
    require_approval: bool
    # router
    route: str
    route_reason: str
    # retrieval / analysis
    retrieval: dict            # {queries_used, coverage_note}
    evidence: list             # [{text, source, distance}]  <- captured from the search tool
    analysis: dict
    # human review
    review_required: bool
    review_reason: str
    review: dict               # {approve: bool, note: str}
    # writing / validation
    report: dict
    validation: dict
    writer_attempts: int
    writer_feedback: str
    # outputs
    direct: dict
    final: dict                # {kind, text, sources, confidence}
    # REDUCER: new entries are APPENDED, not overwritten - this is the audit trail of the run
    steps: Annotated[list, operator.add]


def _step(node: str, t0: float, usage: dict = None, note: str = "") -> list:
    usage = usage or {}
    return [{"node": node, "seconds": round(time.time() - t0, 2), "llm_calls": usage.get("llm_calls", 0),
             "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0), "note": note}]


def build_graph(kb) -> StateGraph:
    """Returns an UNCOMPILED graph. `kb` is captured by closure - the nodes need it for searching."""

    # ------------------------------------------------------------------ nodes
    def router(state: State):
        t0 = time.time()
        mode = state.get("route_mode", config.DEFAULT_ROUTE_MODE)
        rule = logic.rule_route(state["question"])
        forced = logic.apply_route_mode(mode, rule)
        if forced and forced[0]:                                  # decided by code - ZERO LLM calls
            route, reason = forced
            return {"route": route, "route_reason": reason, "steps": _step("router", t0, note=f"{route} ({reason})")}
        decision, usage = agents.run_router(state["question"])    # hard case -> ask the cheap LLM router
        return {"route": decision.route, "route_reason": "llm: " + decision.reason,
                "steps": _step("router", t0, usage, f"{decision.route} (llm)")}

    def direct_rag(state: State):
        t0 = time.time()
        ans, collected, usage = agents.run_direct(state["question"], kb)
        a = ans.model_dump()
        final = {"kind": "answer" if a["confidence"] != "low" else "refusal", "text": a["answer"],
                 "sources": a["sources"], "confidence": a["confidence"]}
        return {"direct": a, "evidence": logic.select_evidence(collected), "final": final,
                "steps": _step("direct_rag", t0, usage, f"confidence {a['confidence']}")}

    def retrieval(state: State):
        t0 = time.time()
        report, evidence, usage = agents.run_retrieval(state["question"], kb)
        return {"retrieval": report.model_dump(), "evidence": evidence,
                "steps": _step("retrieval", t0, usage, f"{len(evidence)} evidence chunks, best distance {logic.best_distance(evidence)}")}

    def refuse(state: State):
        t0 = time.time()
        return {"final": {"kind": "refusal", "text": config.REFUSAL_TEXT, "sources": [], "confidence": "low"},
                "steps": _step("refuse", t0, note="no usable evidence - analyst and writer skipped (saves cost)")}

    def analyst(state: State):
        t0 = time.time()
        analysis, usage = agents.run_analyst(state["question"], state["evidence"])
        a = analysis.model_dump()
        required, reason = logic.needs_review(state["question"], a, state.get("require_approval", config.REQUIRE_APPROVAL))
        return {"analysis": a, "review_required": required, "review_reason": reason,
                "steps": _step("analyst", t0, usage, f"risk {a['risk_level']}, {len(a['conflicts'])} conflicts, {len(a['gaps'])} gaps")}

    def human_review(state: State):
        t0 = time.time()
        # interrupt() PAUSES the whole graph and hands this payload to the caller. When the caller resumes with
        # Command(resume=...), LangGraph re-runs THIS node from the top and interrupt() returns the human's value.
        # => put NO side effects (DB writes, emails) before interrupt().
        decision = interrupt({
            "question": state["question"], "reason": state.get("review_reason", ""),
            "risk_level": state["analysis"]["risk_level"], "risk_reason": state["analysis"]["risk_reason"],
            "key_points": state["analysis"]["key_points"], "conflicts": state["analysis"]["conflicts"],
            "gaps": state["analysis"]["gaps"]})
        return {"review": decision, "steps": _step("human_review", t0, note=f"approve={decision.get('approve')}")}

    def rejected(state: State):
        t0 = time.time()
        note = state.get("review", {}).get("note", "")
        return {"final": {"kind": "rejected", "text": "A reviewer did not approve an automated answer for this question."
                          + (f" Reviewer note: {note}" if note else "") + " Please contact HR directly.",
                          "sources": [], "confidence": "low"},
                "steps": _step("rejected", t0, note="reviewer rejected")}

    def writer(state: State):
        t0 = time.time()
        attempts = state.get("writer_attempts", 0) + 1
        report, usage = agents.run_writer(state["question"], state["analysis"], state["evidence"],
                                          feedback=state.get("writer_feedback", ""),
                                          human_note=state.get("review", {}).get("note", ""))
        return {"report": report.model_dump(), "writer_attempts": attempts,
                "steps": _step("writer", t0, usage, f"attempt {attempts}")}

    def validate(state: State):
        t0 = time.time()
        v = logic.validate_report(state["report"], state["evidence"])
        return {"validation": v, "writer_feedback": v["feedback"],
                "steps": _step("validate", t0, note="passed" if v["passed"] else "failed: " + ", ".join(v["failed"]))}

    def finalize(state: State):
        t0 = time.time()
        r = state["report"]
        return {"final": {"kind": "answer", "text": logic.render_report(r), "sources": logic.all_sources(r),
                          "confidence": r["confidence"]},
                "steps": _step("finalize", t0)}

    def fallback(state: State):
        t0 = time.time()
        return {"final": {"kind": "fallback", "text": config.FALLBACK_TEXT, "sources": [], "confidence": "low"},
                "steps": _step("fallback", t0, note="writer failed validation twice")}

    # ------------------------------------------------------------------ routing functions (CONDITIONAL EDGES)
    def after_router(state: State) -> Literal["direct", "pipeline"]:
        return state["route"]

    def after_retrieval(state: State) -> Literal["refuse", "analyze"]:
        return "analyze" if logic.evidence_is_usable(state.get("evidence", [])) else "refuse"

    def after_analyst(state: State) -> Literal["review", "write"]:
        return "review" if state.get("review_required") else "write"

    def after_review(state: State) -> Literal["approved", "rejected"]:
        return "approved" if state.get("review", {}).get("approve") else "rejected"

    def after_validate(state: State) -> Literal["ok", "retry", "fail"]:
        if state["validation"]["passed"]:
            return "ok"
        return "retry" if state.get("writer_attempts", 0) < config.MAX_WRITER_ATTEMPTS else "fail"

    # ------------------------------------------------------------------ wiring
    g = StateGraph(State)
    for name, fn in [("router", router), ("direct_rag", direct_rag), ("retrieval", retrieval), ("refuse", refuse),
                     ("analyst", analyst), ("human_review", human_review), ("rejected", rejected),
                     ("writer", writer), ("validate", validate), ("finalize", finalize), ("fallback", fallback)]:
        g.add_node(name, fn)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", after_router, {"direct": "direct_rag", "pipeline": "retrieval"})
    g.add_edge("direct_rag", END)
    g.add_conditional_edges("retrieval", after_retrieval, {"refuse": "refuse", "analyze": "analyst"})
    g.add_edge("refuse", END)
    g.add_conditional_edges("analyst", after_analyst, {"review": "human_review", "write": "writer"})
    g.add_conditional_edges("human_review", after_review, {"approved": "writer", "rejected": "rejected"})
    g.add_edge("rejected", END)
    g.add_edge("writer", "validate")
    g.add_conditional_edges("validate", after_validate, {"ok": "finalize", "retry": "writer", "fail": "fallback"})
    g.add_edge("finalize", END)
    g.add_edge("fallback", END)
    return g


def compile_graph(kb):
    """Compile WITH a checkpointer. Without one, interrupt() cannot pause. InMemorySaver forgets on restart;
    a database-backed checkpointer is the production choice."""
    return build_graph(kb).compile(checkpointer=InMemorySaver())
