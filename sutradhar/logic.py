"""PURE-PYTHON decisions. No LLM, no network, no frameworks - so they are cheap, fast and unit-testable.

Rule of thumb for multi-agent design: if a decision can be made by code, make it in code. Use an LLM
only where judgement or language is really needed. Everything in this file is that "code first" layer.
"""
import re

from . import config

# ----------------------------------------------------------------------------- routing
MULTI_MARKERS = ("compare", "difference between", "summar", "across", "all the", "both", "which policies",
                 "what do i need to do", "step by step", "checklist", "implications", "impact", "what are all")


def rule_route(question: str):
    """Cheap first-pass router. Returns (route, reason) or (None, '') when it cannot decide.
    Returning None sends the question to the LLM router (the 'hard' cases)."""
    q = question.lower().strip()
    words = len(q.split())
    if any(m in q for m in MULTI_MARKERS):
        return "pipeline", "rule: multi-part marker words in the question"
    if q.count("?") > 1 or q.count(" and ") >= 2 or words > config.LONG_QUESTION_WORDS:
        return "pipeline", "rule: long or multi-part question"
    if words <= config.SHORT_QUESTION_WORDS and " and " not in q:
        return "direct", "rule: short single-fact question"
    return None, ""


def apply_route_mode(mode: str, rule_result):
    """Experiment switch used by the UI and compare_routes.py."""
    if mode == "always_direct":
        return "direct", "forced by route mode"
    if mode == "always_pipeline":
        return "pipeline", "forced by route mode"
    return rule_result


# ----------------------------------------------------------------------------- evidence
def select_evidence(collected: list, max_n: int = config.MAX_EVIDENCE) -> list:
    """Dedupe raw search hits and keep the closest ones. `collected` items are dicts with
    text/source/distance, captured from the search tool (never typed by an LLM)."""
    seen, unique = set(), []
    for hit in sorted(collected, key=lambda h: h["distance"]):
        key = (hit["source"], hit["text"])
        if key not in seen:
            seen.add(key)
            unique.append(hit)
    return unique[:max_n]


def best_distance(evidence: list) -> float:
    return min((e["distance"] for e in evidence), default=9.9)


def evidence_is_usable(evidence: list) -> bool:
    return bool(evidence) and best_distance(evidence) <= config.WEAK_MATCH


def format_evidence(evidence: list) -> str:
    return "\n\n".join(f"[E{i}] source: {e['source']} | distance {e['distance']}\n{e['text']}"
                       for i, e in enumerate(evidence, 1))


# ----------------------------------------------------------------------------- human-in-the-loop
def needs_review(question: str, analysis: dict, require: bool = True):
    """Should a human approve before the writer runs? Returns (bool, reason)."""
    if not require:
        return False, ""
    q = question.lower()
    for topic in config.HIGH_RISK_TOPICS:
        if topic in q:
            return True, f"question mentions '{topic}'"
    if analysis.get("risk_level") == "high":
        return True, "analyst rated risk HIGH: " + analysis.get("risk_reason", "")
    if analysis.get("conflicts"):
        return True, "analyst found conflicting sources"
    return False, ""


# ----------------------------------------------------------------------------- validation
PII_PATTERNS = [re.compile(r"(?:\+91[-\s]?)?[6-9]\d{9}\b"), re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")]


def validate_report(report: dict, evidence: list) -> dict:
    """Deterministic checks on the writer's output - the final gate before the user sees anything."""
    allowed = {e["source"] for e in evidence}
    sections = report.get("sections", [])
    cited = [s for sec in sections for s in sec.get("sources", [])]
    text = " ".join([report.get("summary", "")] + [sec.get("body", "") for sec in sections])
    checks = {
        "has_content": bool(report.get("summary", "").strip()) and len(sections) > 0,
        "every_section_cited": all(sec.get("sources") for sec in sections) and len(sections) > 0,
        "citations_exist_in_evidence": all(c in allowed for c in cited) and bool(cited),
        "no_pii": not any(p.search(text) for p in PII_PATTERNS),
    }
    failed = [k for k, ok in checks.items() if not ok]
    feedback = ""
    if failed:
        feedback = ("Fix these problems: " + ", ".join(failed) + ". Allowed source files are: "
                    + ", ".join(sorted(allowed)) + ". Every section must list at least one allowed source.")
    return {"passed": not failed, "checks": checks, "failed": failed, "feedback": feedback}


def all_sources(report: dict) -> list:
    out = []
    for sec in report.get("sections", []):
        for s in sec.get("sources", []):
            if s not in out:
                out.append(s)
    return out


def render_report(report: dict) -> str:
    lines = [f"### {report['title']}", report["summary"], ""]
    for sec in report["sections"]:
        lines += [f"**{sec['heading']}**", sec["body"], f"*(Source: {', '.join(sec['sources'])})*", ""]
    if report.get("caveats"):
        lines.append("**Caveats**")
        lines += [f"- {c}" for c in report["caveats"]]
    return "\n".join(lines).strip()
