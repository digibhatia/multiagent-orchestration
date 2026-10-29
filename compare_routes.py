"""Same questions, three route modes: always_direct vs always_pipeline vs auto. Prints cost and latency.

    python compare_routes.py

Human review is switched OFF here so the runs never pause. Costs real API calls (about 3 x 5 questions).
Read the answers yourself too: cheaper is only better if the answer is still good enough.
"""
from sutradhar import runtime
from sutradhar.graph import compile_graph
from sutradhar.kb import KnowledgeBase

QUESTIONS = [
    "How many casual leave days do I get in a year?",
    "How many days a week can I work from home?",
    "Compare the work-from-home and leave policies for a new joiner and summarise what I must do in my first week.",
    "My laptop was stolen while I was travelling for work. What do the IT, laptop and travel policies say I must do?",
    "What is the reimbursement limit for domestic hotel stays and do I need approval for travel?",
]

graph = compile_graph(KnowledgeBase())
rows = []
for mode in ("always_direct", "always_pipeline", "auto"):
    calls = tin = tout = 0
    secs = 0.0
    routes = []
    for q in QUESTIONS:
        out = runtime.start(graph, q, route_mode=mode, require_approval=False)
        st = out["state"]
        t = runtime.totals(st["steps"])
        calls += t["llm_calls"]; tin += t["input_tokens"]; tout += t["output_tokens"]; secs += t["seconds"]
        routes.append(st["route"][0].upper())          # D or P
    rows.append((mode, calls, tin + tout, round(secs / len(QUESTIONS), 1), "".join(routes)))

print(f"\n{'mode':<17}{'LLM calls':>10}{'total tokens':>14}{'avg seconds':>13}   routes (D=direct P=pipeline)")
for r in rows:
    print(f"{r[0]:<17}{r[1]:>10}{r[2]:>14}{r[3]:>13}   {r[4]}")
print("\nDiscuss: which questions did NOT need the pipeline? What did the pipeline answers add that direct answers lacked?")
