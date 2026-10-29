# Sutradhar — Multi-Agent Enterprise Assistant (Day 10: Multi-Agent Systems & Orchestration)

Sutradhar ("the one who holds the strings" — the conductor of a play) upgrades the Day 9 assistant from **one agent** to a
**team of agents** coordinated by a **LangGraph** state machine:

```
question ──► ROUTER ──► direct RAG answer                                   (simple, one-fact questions)
                 └───► RETRIEVAL agent ─► ANALYST agent ─► [human review] ─► WRITER agent ─► validation ─► cited report
```

Built with **LangGraph** (orchestration), **PydanticAI** (the individual agents), **ChromaDB + OpenAI embeddings** (the RAG from Modules 5–6),
and **Streamlit** (UI). CrewAI and AutoGen are covered for comparison in the guide document, not run here.

---

## 1. What you will learn by running this

| Concept | Where to see it |
|---|---|
| Orchestrator vs worker agents | `graph.py` (orchestrator) vs `agents.py` (workers) |
| Agent-to-agent hand-off with typed contracts | `schemas.py` — every hand-off is a Pydantic model |
| LangGraph: state, nodes, edges | `class State`, `add_node`, `add_edge` in `graph.py` |
| Conditional routing | five `after_*` functions + `add_conditional_edges` |
| Routing layer: code first, LLM second | `logic.rule_route()` then `agents.run_router()` |
| Human-in-the-loop checkpoint | `human_review` node, `interrupt()`, `runtime.resume()`, the approve/reject panel in the UI |
| Stateful graphs and checkpointing | `InMemorySaver`, `thread_id`, `graph.get_state()` |
| Evidence captured from tools, not typed by an LLM | `KBDeps.collected` in `agents.py` |
| Validation loop with retry and fallback | `validate` → `writer` retry → `fallback` |
| Cost / latency / quality trade-off of multi-agent | `compare_routes.py`, the "Node timeline" in the UI |

## 2. Architecture flow

```
                    ┌────────────────────────── Streamlit UI (app.py) ───────────────────────────┐
  question ───────► │ route mode · approval toggle · node timeline · evidence · analysis · report │
                    └───────────────┬───────────────────────────────────────────▲─────────────────┘
                                    │ runtime.start()/resume()                  │ state snapshot
                                    ▼                                           │
   ┌─────────────────────────── LangGraph (graph.py) + InMemorySaver checkpointer ─────────────────┐
   │ router ──(rules: 0 LLM calls)──┐                                                                │
   │        └─(LLM router if unsure)┴─► direct_rag ───────────────────────────────────────► END     │
   │                                └─► retrieval ──► (no usable evidence) ─► refuse ───────► END     │
   │                                         └─► analyst ──► (risky?) ─► human_review ─┬─► rejected ─► END
   │                                                  └────────────────────────────────┴─► writer ◄─┐│
   │                                                                                         ▼      ││
   │                                                                            validate ──retry───┘│
   │                                                                              ok │     fail │   │
   │                                                                          finalize ► END  fallback ► END
   └───────────────────────────────────────────────────────────────────────────────────────────────┘
        workers (agents.py, PydanticAI):  router · direct · retrieval(+search tool) · analyst · writer
        RAG (kb.py): ChromaDB + text-embedding-3-small          Pure code (logic.py): routing rules, risk rules, validation
```

Key design ideas:

1. **Code first, LLM second.** If Python can decide (is it short? does it contain "compare"? are the citations real?), Python decides — free, instant, testable.
2. **Typed hand-offs.** Retrieval → Analyst → Writer exchange validated models, never loose text.
3. **Evidence is captured from the search tool**, so no agent can invent or alter a quote. The writer may only cite files that really appeared in the evidence.
4. **Pause for a human on risk.** Risky topics or a high-risk analysis stop the graph until a reviewer approves or rejects.
5. **Cheap exits.** No usable evidence → refuse immediately; analyst and writer never run.

## 3. Setup

```bash
cd sutradhar_multiagent
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # Windows: copy .env.example .env
# open .env and set:  OPENAI_API_KEY="sk-..."
```

> **Model names.** Defaults are `gpt-5.4-mini` for both the router and the workers (OpenAI retires older models). The curriculum model is
> GPT-4o — if your account still serves it, set `MODEL_ROUTER` / `MODEL_WORKERS` in `.env`. Using a cheaper router than the workers is the usual pattern.

**Test with no API key** (our wiring on a tiny fake LangGraph + fake agents, plus pure logic):

```bash
python selftest_offline.py           # must end with: All 26 offline checks passed.
```

> This proves *our* routing keys, edges and resume logic are consistent. It does **not** prove real LangGraph behaves the same — the real run below does.

## 4. Run it — two terminals

**Step 1 — build the knowledge base (once):**
```bash
python build_vectorstore.py          # embeds 7 policy documents into chroma_db/
python inspect_kb.py "how many casual leave days do I get?"
```

**Terminal 1 — the UI:**
```bash
streamlit run app.py                 # http://localhost:8501
```

**Terminal 2 — scripted questions through the same graph:**
```bash
python run_demo.py                   # asks you to approve/reject when the graph pauses
python run_demo.py --approve yes     # auto-approve
```

**Trade-off experiment (about 15 real runs):**
```bash
python compare_routes.py             # always_direct vs always_pipeline vs auto
```

## 5. Guided exercises

1. **Two paths.** In the UI ask *"How many casual leave days do I get in a year?"* then *"Compare the work-from-home and leave policies for a new joiner and summarise what I must do in my first week."* Compare **Route**, **LLM calls**, **Seconds** and the Node timeline.
2. **Who routed?** Look at the router row of each timeline. One question used 0 LLM calls, which one, and why? Ask the 19-word question in `run_demo.py` ("router judgement") and see when the LLM router is used.
3. **Force the paths.** Set Route mode to `always_pipeline`, ask the short question; then `always_direct` for the multi-part one. What do you gain or lose?
4. **Pause and resume.** Ask *"What are all the consequences under the code of conduct if an employee is found guilty of misconduct…"*. The graph pauses. Read the analyst's reason, add a note to the writer, approve. Repeat and reject.
5. **Nothing found.** Ask *"What is the company's stock price target?"* Which nodes ran? Which did not? Why is that cheaper?
6. **Trust the evidence.** Open "Evidence captured from the search tool". Compare it with the cited sources in the report. Why does `validate` compare the two?
7. **Break validation.** In `agents.py` change the writer prompt so it may cite any file name it likes. Re-run a pipeline question — what does the timeline show (writer ×2, fallback)? Change it back.
8. **Cost view.** Run `compare_routes.py`. Which mode is cheapest? Which questions truly needed the pipeline?

## 6. What to notice

- **Node timeline** is the audit trail of the run (a `steps` list that every node *appends* to via a reducer).
- **Router row:** rules cost nothing; the LLM router only appears for questions the rules cannot decide.
- **Retrieval row note:** the number of evidence chunks and the best distance — the refusal decision uses the same numbers as Day 9.
- **Pipeline = more LLM calls** (retrieval + analyst + writer, sometimes writer twice). Quality gains must justify this.
- **Human review resumes the node from its top.** That is why nothing with side effects may come before `interrupt()`.
- **The checkpointer is what makes pause/resume possible.** `InMemorySaver` forgets on restart; use a database-backed saver in production.
- **`thread_id`** isolates one question's state from another's.

## 7. Trade-offs cheat-sheet

| | Direct RAG | Full pipeline |
|---|---|---|
| LLM calls | 1–3 | 4–7 |
| Latency | lowest | highest |
| Best for | one fact, one policy | multi-document, comparisons, summaries, risky topics |
| Auditability | answer + sources | evidence → analysis → report, each inspectable |
| Human checkpoint | none | yes, on risk |
| Failure mode | misses part of a multi-part question | cost and latency; errors can compound between agents |

## 8. Folder map

```
sutradhar_multiagent/
├── app.py                 Streamlit UI incl. human-review panel      (Terminal 1)
├── run_demo.py            scripted questions, console pause/resume   (Terminal 2)
├── compare_routes.py      direct vs pipeline vs auto cost table
├── selftest_offline.py    26 offline checks (no key, no network)
├── build_vectorstore.py   chunk + embed documents into ChromaDB
├── inspect_kb.py          see raw RAG results and distances
├── knowledge_base/        7 sample policy documents (5 HR, 2 IT)
├── offline_tests/         fake LangGraph stand-in used only by selftest_offline.py
└── sutradhar/
    ├── config.py          ALL tunables
    ├── schemas.py         typed hand-offs between agents
    ├── logic.py           pure-code routing, review and validation rules
    ├── agents.py          PydanticAI worker agents (router, direct, retrieval, analyst, writer)
    ├── graph.py           LangGraph orchestrator (State, nodes, conditional edges)
    ├── runtime.py         start() / resume() helpers around the graph
    └── kb.py              RAG: chunk, embed, search
```

## 9. Known limits (be honest about these)

- Written against **LangGraph ≥ 0.6** and **PydanticAI 1.x**. Both move quickly; if an import or parameter name differs in your version, check the release notes. Interrupts need a checkpointer; the `__interrupt__` key in `invoke()` results needs a recent LangGraph (the code falls back to `get_state()`).
- `InMemorySaver` loses paused runs when the server restarts. The Streamlit app caches one graph per server process so pauses survive reruns, not restarts.
- The router rules are deliberately simple word lists; they are a teaching device, not a classifier. Measure them on a labelled set (see the assignment).
- Risk detection uses a keyword list plus the analyst's judgement. It will miss things; the human checkpoint is a safety net, not a guarantee.
- No conversation memory in this repo: each question is its own thread (Day 9's memory can be plugged in as an extension).
- Documents are sample training data.

## 10. Take-home

See the separate unguided assignment outline (Word document).
