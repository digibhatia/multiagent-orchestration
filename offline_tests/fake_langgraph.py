"""A TINY STAND-IN for LangGraph so we can test OUR wiring (nodes, edges, routing keys, interrupt/resume)
without installing anything. It mimics only the pieces Sutradhar uses. It is NOT LangGraph - always run the
real thing (python run_demo.py) before trusting the graph in class."""
import sys
import types
from typing import Annotated, get_args, get_origin, get_type_hints

START, END = "__start__", "__end__"
_NONE = object()
_resume = _NONE


class _Pause(Exception):
    def __init__(self, payload):
        self.payload = payload


class Interrupt:
    def __init__(self, value):
        self.value = value


class Command:
    def __init__(self, resume=None):
        self.resume = resume


def interrupt(payload):
    global _resume
    if _resume is not _NONE:
        v, _resume = _resume, _NONE
        return v
    raise _Pause(payload)


class InMemorySaver:
    def __init__(self):
        self.data = {}


class _Snap:
    def __init__(self, values, nxt):
        self.values, self.next, self.tasks = values, nxt, []


class StateGraph:
    def __init__(self, schema):
        self.reducers = {}
        for k, t in get_type_hints(schema, include_extras=True).items():
            if get_origin(t) is Annotated:
                self.reducers[k] = get_args(t)[1]
        self.nodes, self.edges, self.cond = {}, {}, {}

    def add_node(self, name, fn): self.nodes[name] = fn
    def add_edge(self, a, b): self.edges[a] = b

    def add_conditional_edges(self, src, fn, path_map=None):
        self.cond[src] = (fn, path_map)

    def compile(self, checkpointer=None):
        return _Compiled(self, checkpointer or InMemorySaver())


class _Compiled:
    def __init__(self, g, cp):
        self.g, self.cp = g, cp

    def _apply(self, values, update):
        for k, v in (update or {}).items():
            values[k] = self.g.reducers[k](values.get(k, []), v) if k in self.g.reducers else v

    def _next(self, node, values):
        if node in self.g.cond:
            fn, pm = self.g.cond[node]
            key = fn(values)
            assert (pm is None) or key in pm, f"router function for '{node}' returned unknown key {key!r}"
            return pm[key] if pm else key
        return self.g.edges[node]

    def invoke(self, inp, config):
        global _resume
        tid = config["configurable"]["thread_id"]
        store = self.cp.data.setdefault(tid, {"values": {}, "next": None})
        if isinstance(inp, Command):
            node, _resume = store["next"], inp.resume
            assert node, "nothing to resume"
        else:
            store["values"] = {}
            self._apply(store["values"], inp)
            node = self.g.edges[START]
        values = store["values"]
        while node != END:
            try:
                update = self.g.nodes[node](values)
            except _Pause as p:
                store["next"] = node
                return {**values, "__interrupt__": [Interrupt(p.payload)]}
            self._apply(values, update)
            node = self._next(node, values)
        store["next"] = None
        return dict(values)

    def get_state(self, config):
        s = self.cp.data[config["configurable"]["thread_id"]]
        return _Snap(dict(s["values"]), (s["next"],) if s["next"] else ())


def install():
    """Register fake modules under the real import names."""
    lg = types.ModuleType("langgraph")
    graph = types.ModuleType("langgraph.graph"); graph.StateGraph, graph.START, graph.END = StateGraph, START, END
    typ = types.ModuleType("langgraph.types"); typ.interrupt, typ.Command = interrupt, Command
    cp = types.ModuleType("langgraph.checkpoint"); mem = types.ModuleType("langgraph.checkpoint.memory"); mem.InMemorySaver = InMemorySaver
    pai = types.ModuleType("pydantic_ai")

    class _Agent:
        def __init__(self, *a, **k): pass
        def tool(self, fn): return fn
        def tool_plain(self, fn): return fn
        def instructions(self, fn): return fn

    class _RC:
        def __class_getitem__(cls, item): return cls

    pai.Agent, pai.RunContext = _Agent, _RC
    usage = types.ModuleType("pydantic_ai.usage"); usage.UsageLimits = lambda **k: None
    for name, m in [("langgraph", lg), ("langgraph.graph", graph), ("langgraph.types", typ), ("langgraph.checkpoint", cp),
                    ("langgraph.checkpoint.memory", mem), ("pydantic_ai", pai), ("pydantic_ai.usage", usage)]:
        sys.modules[name] = m
