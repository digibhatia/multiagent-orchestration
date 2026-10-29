"""OFFLINE tests - no API key, no network, no LangGraph/PydanticAI needed.

    python selftest_offline.py

Part 1 tests the pure decision logic (routing rules, review rules, validation).
Part 2 runs OUR graph on a tiny fake LangGraph with fake agents, checking every path:
direct, pipeline, refuse, human approve, human reject, writer retry, writer fallback, LLM-router case.
It proves our wiring is consistent. It does NOT prove the real LangGraph behaves the same - run run_demo.py for that.
"""
import sys

from offline_tests import fake_langgraph

fake_langgraph.install()          # must happen BEFORE importing sutradhar.graph / agents

from sutradhar import agents, config, logic, runtime  # noqa: E402
from sutradhar.graph import compile_graph             # noqa: E402
from sutradhar.schemas import Analysis, DirectAnswer, KeyPoint, Report, ReportSection, RetrievalReport, RouteDecision  # noqa: E402

ok = 0


def check(name, cond):
    global ok
    assert cond, f"FAILED: {name}"
    ok += 1
    print(f"  ok  {name}")


print("PART 1 - pure logic")
check("short single fact -> direct", logic.rule_route("How many casual leave days do I get?")[0] == "direct")
check("'compare' -> pipeline", logic.rule_route("Compare WFH and leave policy")[0] == "pipeline")
check("two questions -> pipeline", logic.rule_route("What is CL? What is SL?")[0] == "pipeline")
check("middle length, no marker -> undecided (LLM router)",
      logic.rule_route("Explain how the performance review cycle works for someone who joined the company in the middle of the year")[0] is None)
check("always_pipeline overrides", logic.apply_route_mode("always_pipeline", ("direct", "x"))[0] == "pipeline")
ev = [{"text": "a", "source": "01.txt", "distance": 0.4}, {"text": "a", "source": "01.txt", "distance": 0.5},
      {"text": "b", "source": "02.txt", "distance": 0.3}]
sel = logic.select_evidence(ev)
check("evidence deduped and sorted", len(sel) == 2 and sel[0]["source"] == "02.txt")
check("weak evidence unusable", not logic.evidence_is_usable([{"text": "z", "source": "x", "distance": 0.9}]))
check("review: risky topic word", logic.needs_review("what if I am fired", {})[0])
check("review: analyst high risk", logic.needs_review("hello there", {"risk_level": "high", "risk_reason": "r"})[0])
check("review: conflicts", logic.needs_review("hello there", {"risk_level": "low", "conflicts": ["x"]})[0])
check("review: off switch", not logic.needs_review("what if I am fired", {}, require=False)[0])
good = {"summary": "s", "sections": [{"heading": "h", "body": "b", "sources": ["01.txt"]}]}
check("validate good report", logic.validate_report(good, sel[:1] and [{"source": "01.txt"}])["passed"])
bad = {"summary": "s", "sections": [{"heading": "h", "body": "call 9988776655", "sources": ["invented.txt"]}]}
v = logic.validate_report(bad, [{"source": "01.txt"}])
check("validate catches invented source + PII", {"citations_exist_in_evidence", "no_pii"} <= set(v["failed"]))

print("\nPART 2 - graph wiring with fake agents")
EVID = [{"text": "Casual leave is 12 days.", "source": "01_leave_policy.txt", "distance": 0.4},
        {"text": "WFH is 2 days.", "source": "02_wfh_hybrid_policy.txt", "distance": 0.45}]
calls = {"router": 0, "analyst": 0, "writer": 0}
state = {"evidence": EVID, "risk": "low", "writer_sources": ["01_leave_policy.txt"], "bad_writes": 0}


def fake_router(q):
    calls["router"] += 1
    return RouteDecision(route="pipeline", reason="needs two docs"), {"llm_calls": 1, "input_tokens": 10, "output_tokens": 5}


def fake_direct(q, kb):
    return DirectAnswer(answer="12 days", sources=["01_leave_policy.txt"], confidence="high"), EVID, {"llm_calls": 2, "input_tokens": 50, "output_tokens": 9}


def fake_retrieval(q, kb):
    return RetrievalReport(queries_used=["a", "b"], coverage_note="ok"), state["evidence"], {"llm_calls": 3, "input_tokens": 80, "output_tokens": 20}


def fake_analyst(q, ev):
    calls["analyst"] += 1
    return Analysis(key_points=[KeyPoint(point="p", sources=["01_leave_policy.txt"])], risk_level=state["risk"], risk_reason="because"), \
        {"llm_calls": 1, "input_tokens": 60, "output_tokens": 30}


def fake_writer(q, analysis, ev, feedback="", human_note=""):
    calls["writer"] += 1
    srcs = state["writer_sources"] if state["bad_writes"] > 0 else ["01_leave_policy.txt"]
    if state["bad_writes"] > 0:
        state["bad_writes"] -= 1
    return Report(title="T", summary="S", sections=[ReportSection(heading="H", body="B", sources=srcs)], confidence="high"), \
        {"llm_calls": 1, "input_tokens": 70, "output_tokens": 40}


agents.run_router, agents.run_direct, agents.run_retrieval = fake_router, fake_direct, fake_retrieval
agents.run_analyst, agents.run_writer = fake_analyst, fake_writer
graph = compile_graph(kb=None)


def reset(**kw):
    state.update({"evidence": EVID, "risk": "low", "writer_sources": ["01_leave_policy.txt"], "bad_writes": 0}); state.update(kw)
    for k in calls: calls[k] = 0


def nodes(out): return [s["node"] for s in out["state"]["steps"]]


reset()
o = runtime.start(graph, "How many casual leave days do I get?")
check("direct path nodes", nodes(o) == ["router", "direct_rag"] and o["state"]["final"]["kind"] == "answer")
check("rule router made ZERO llm calls", calls["router"] == 0 and o["state"]["steps"][0]["llm_calls"] == 0)

reset()
o = runtime.start(graph, "Compare the wfh and leave policies for a new joiner", require_approval=False)
check("pipeline nodes", nodes(o) == ["router", "retrieval", "analyst", "writer", "validate", "finalize"])
check("final report has sources", o["state"]["final"]["sources"] == ["01_leave_policy.txt"])
t = runtime.totals(o["state"]["steps"])
check("totals add up", t["llm_calls"] == 0 + 3 + 1 + 1)

reset(evidence=[])
o = runtime.start(graph, "Compare the wfh and leave policies", require_approval=False)
check("refuse path skips analyst+writer", nodes(o) == ["router", "retrieval", "refuse"] and calls["analyst"] == 0 and o["state"]["final"]["kind"] == "refusal")

reset(risk="high")
o = runtime.start(graph, "Compare the wfh and leave policies for new joiners")
check("pauses at human review", o["status"] == "waiting" and o["interrupt"]["risk_level"] == "high")
tid = o["thread_id"]
o2 = runtime.resume(graph, tid, True, "mention appeals")
check("approve continues to finalize", o2["status"] == "done" and nodes(o2)[-1] == "finalize" and "human_review" in nodes(o2))

reset(risk="high")
o = runtime.start(graph, "Compare the wfh and leave policies for new joiners")
o2 = runtime.resume(graph, o["thread_id"], False, "no")
check("reject ends with rejected", o2["state"]["final"]["kind"] == "rejected" and calls["writer"] == 0)

reset(bad_writes=1, writer_sources=["invented.txt"])
o = runtime.start(graph, "Compare the wfh and leave policies", require_approval=False)
check("writer retry then ok", nodes(o).count("writer") == 2 and o["state"]["final"]["kind"] == "answer" and o["state"]["writer_attempts"] == 2)

reset(bad_writes=5, writer_sources=["invented.txt"])
o = runtime.start(graph, "Compare the wfh and leave policies", require_approval=False)
check("writer fails twice -> fallback", o["state"]["final"]["kind"] == "fallback" and nodes(o).count("writer") == config.MAX_WRITER_ATTEMPTS)

reset()
o = runtime.start(graph, "Explain how the performance review cycle works for someone who joined the company in the middle of the year", require_approval=False)
check("ambiguous question uses LLM router", calls["router"] == 1 and o["state"]["route"] == "pipeline")

reset()
o = runtime.start(graph, "How many casual leave days do I get?", route_mode="always_pipeline", require_approval=False)
check("route mode override forces pipeline", o["state"]["route"] == "pipeline" and "retrieval" in nodes(o))

print(f"\nAll {ok} offline checks passed.")
