"""STEP 2 (Terminal 1) - the Sutradhar UI.

    streamlit run app.py

Ask a question. You will SEE: which route the router chose and why, every node that ran (time, LLM calls, tokens),
the evidence the retrieval agent captured, the analyst's findings, and the writer's cited report.
If the analyst flags risk, the graph PAUSES here and asks YOU (the human reviewer) to approve or reject.
"""
import streamlit as st

from sutradhar import config, runtime
from sutradhar.graph import compile_graph
from sutradhar.kb import KnowledgeBase

st.set_page_config(page_title="Sutradhar", page_icon="🧭", layout="wide")

EXAMPLES = [
    "How many casual leave days do I get in a year?",
    "Compare the work-from-home and leave policies for a new joiner and summarise what I must do in my first week.",
    "What are all the consequences under the code of conduct if an employee is found guilty of misconduct, and what is the appeal process?",
    "My laptop was stolen while I was travelling for work. What do the IT, laptop and travel policies say I must do?",
    "What is the company's stock price target?",
]


@st.cache_resource  # one graph + one checkpointer for the whole server, so paused runs survive Streamlit reruns
def get_graph():
    kb = KnowledgeBase()
    return compile_graph(kb)


graph = get_graph()
S = st.session_state
S.setdefault("history", [])     # finished runs
S.setdefault("pending", None)   # a run paused at human_review

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🧭 Sutradhar")
route_mode = st.sidebar.selectbox("Route mode", ["auto", "always_direct", "always_pipeline"],
                                  help="auto = router decides. The other two force a path so you can compare cost and quality.")
require_approval = st.sidebar.checkbox("Human approval for risky answers", value=config.REQUIRE_APPROVAL)
st.sidebar.caption(f"Router model: {config.MODEL_ROUTER} · Worker model: {config.MODEL_WORKERS}")
st.sidebar.markdown("**Try an example**")
for i, ex in enumerate(EXAMPLES):
    if st.sidebar.button(ex[:60] + ("…" if len(ex) > 60 else ""), key=f"ex{i}", disabled=S.pending is not None):
        S["queued"] = ex
if st.sidebar.button("Clear history"):
    S.history, S.pending = [], None
    st.rerun()

st.title("Sutradhar — multi-agent enterprise assistant")
st.caption("Router → (direct RAG) or (retrieval → analyst → [human review] → writer → validation)")


def show_run(run: dict):
    st.markdown(f"#### ❓ {run['question']}")
    stt = run["state"]
    tot = runtime.totals(stt.get("steps", []))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", stt.get("route", "-"))
    c2.metric("LLM calls", tot["llm_calls"])
    c3.metric("Tokens (in/out)", f"{tot['input_tokens']}/{tot['output_tokens']}")
    c4.metric("Seconds", tot["seconds"])
    st.caption(f"Why this route: {stt.get('route_reason', '-')}")
    final = stt.get("final")
    if final:
        {"answer": st.success, "refusal": st.warning, "rejected": st.error, "fallback": st.error}.get(final["kind"], st.info)(
            f"**{final['kind'].upper()}** · confidence: {final['confidence']}")
        st.markdown(final["text"])
        if final["sources"]:
            st.caption("Sources: " + ", ".join(final["sources"]))
    with st.expander("🔁 Node timeline (what ran, in order)"):
        st.dataframe(stt.get("steps", []), use_container_width=True)
    if stt.get("retrieval"):
        with st.expander("🔎 Retrieval agent — queries and coverage"):
            st.json(stt["retrieval"])
    if stt.get("evidence"):
        with st.expander(f"📄 Evidence captured from the search tool ({len(stt['evidence'])})"):
            st.dataframe([{"source": e["source"], "distance": e["distance"], "text": e["text"][:160] + "…"} for e in stt["evidence"]],
                         use_container_width=True)
    if stt.get("analysis"):
        with st.expander("🧠 Analyst agent output"):
            st.json(stt["analysis"])
    if stt.get("review"):
        with st.expander("🧑‍⚖️ Human review decision"):
            st.json({"reason": stt.get("review_reason"), **stt["review"]})
    if stt.get("validation"):
        with st.expander("✅ Validation checks"):
            st.json(stt["validation"])
    st.divider()


# ------------------------------------------------------------------ human-in-the-loop panel
if S.pending:
    p = S.pending
    st.warning("⏸ The graph is PAUSED at human_review and is waiting for you.")
    info = p["interrupt"]
    st.markdown(f"**Question:** {info['question']}")
    st.markdown(f"**Why review is needed:** {info['reason']}")
    st.markdown(f"**Analyst risk:** {info['risk_level']} — {info['risk_reason']}")
    with st.expander("Analyst key points, conflicts and gaps", expanded=True):
        st.json({k: info[k] for k in ("key_points", "conflicts", "gaps")})
    note = st.text_input("Optional note to the writer (e.g. 'mention the appeal window')", key="note")
    a, b = st.columns(2)
    if a.button("✅ Approve and continue", type="primary"):
        out = runtime.resume(graph, p["thread_id"], True, note)
        S.pending = None
        S.history.append({"question": info["question"], **out})
        st.rerun()
    if b.button("⛔ Reject"):
        out = runtime.resume(graph, p["thread_id"], False, note)
        S.pending = None
        S.history.append({"question": info["question"], **out})
        st.rerun()
    st.stop()

# ------------------------------------------------------------------ new question
q = st.chat_input("Ask a policy question…") or S.pop("queued", None)
if q:
    with st.spinner("Agents working…"):
        out = runtime.start(graph, q, route_mode=route_mode, require_approval=require_approval)
    if out["status"] == "waiting":
        S.pending = {**out, "question": q}
    else:
        S.history.append({"question": q, **out})
    st.rerun()

for run in reversed(S.history):
    show_run(run)
if not S.history:
    st.info("Pick an example in the sidebar, or type a question. Try one short question and one multi-part question and compare the timelines.")
