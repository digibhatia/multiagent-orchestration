"""STEP 3 (Terminal 2) - run a scripted set of questions through the SAME graph from the command line.

    python run_demo.py                 # asks you to approve/reject when the graph pauses
    python run_demo.py --approve yes   # auto-approve every pause
    python run_demo.py --approve no    # auto-reject every pause

Edit QUESTIONS, re-run, and compare routes. Each question exercises a different path through the graph.
"""
import argparse

from sutradhar import runtime
from sutradhar.graph import compile_graph
from sutradhar.kb import KnowledgeBase

QUESTIONS = [
    ("direct path",        "How many casual leave days do I get in a year?"),
    ("pipeline path",      "Compare the work-from-home and leave policies for a new joiner and summarise what I must do in my first week."),
    ("human review",       "What are all the consequences under the code of conduct if an employee is found guilty of misconduct, and what is the appeal process?"),
    ("multi-document",     "My laptop was stolen while I was travelling for work. What do the IT, laptop and travel policies say I must do?"),
    ("refusal",            "What is the company's stock price target and what bonus will the CEO get next year?"),
    ("router judgement",   "Explain how the performance review cycle works for someone who joined the company in the middle of the year"),   # 19 words, no marker words: the rules cannot decide, so the LLM router is used
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve", choices=["ask", "yes", "no"], default="ask")
    args = ap.parse_args()
    graph = compile_graph(KnowledgeBase())

    for label, q in QUESTIONS:
        print("=" * 78, f"\n[{label}]  {q}")
        out = runtime.start(graph, q)
        if out["status"] == "waiting":
            info = out["interrupt"]
            print(f"\n  ⏸  PAUSED for human review: {info['reason']}")
            print(f"     risk: {info['risk_level']} - {info['risk_reason']}")
            decision = args.approve
            if decision == "ask":
                decision = "yes" if input("     approve? [y/N] ").strip().lower().startswith("y") else "no"
            out = runtime.resume(graph, out["thread_id"], decision == "yes", note="")
        st = out["state"]
        print(f"\n  route: {st['route']}  ({st['route_reason']})")
        for s in st["steps"]:
            print(f"   - {s['node']:<13} {s['seconds']:>5}s  calls={s['llm_calls']}  tok={s['input_tokens']}/{s['output_tokens']}  {s['note']}")
        t = runtime.totals(st["steps"])
        print(f"  TOTAL: {t['llm_calls']} LLM calls, {t['input_tokens']}/{t['output_tokens']} tokens, {t['seconds']}s")
        f = st["final"]
        print(f"\n  [{f['kind'].upper()} | confidence {f['confidence']} | sources {f['sources']}]\n{f['text']}\n")


if __name__ == "__main__":
    main()
