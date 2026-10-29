"""TYPED CONTRACTS between agents.

In a multi-agent system the most common bug is one agent handing the next a vague blob of text.
Here every hand-off is a Pydantic model: the framework validates it, so a malformed hand-off fails
loudly at the boundary instead of silently corrupting the next agent's work.
"""
from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]


class RouteDecision(BaseModel):
    """Router output."""
    route: Literal["direct", "pipeline"] = Field(
        description="direct = one fact from one policy; pipeline = multi-part, compares documents, or needs a structured write-up")
    reason: str = Field(description="One short sentence explaining the choice.")


class RetrievalReport(BaseModel):
    """Retrieval agent output. NOTE: the evidence itself is NOT here - it is captured from the tool
    calls (see agents.py) so the LLM can never invent or alter a quote."""
    queries_used: list[str] = Field(description="The search queries you ran, one per sub-topic.")
    coverage_note: str = Field(description="Which parts of the question the results cover and which they do not.")


class KeyPoint(BaseModel):
    point: str = Field(description="One factual statement taken from the evidence.")
    sources: list[str] = Field(description="File names (e.g. 01_leave_policy.txt) that support the point.")


class Analysis(BaseModel):
    """Analyst agent output."""
    key_points: list[KeyPoint]
    conflicts: list[str] = Field(default_factory=list, description="Places where sources disagree. Empty if none.")
    gaps: list[str] = Field(default_factory=list, description="Parts of the question the evidence does not answer.")
    risk_level: Literal["low", "medium", "high"] = Field(
        description="high = answer affects discipline, legal exposure, security incidents or money; low = routine information.")
    risk_reason: str


class ReportSection(BaseModel):
    heading: str
    body: str
    sources: list[str] = Field(description="File names used for this section; must come from the evidence.")


class Report(BaseModel):
    """Writer agent output."""
    title: str
    summary: str = Field(description="Two sentences maximum.")
    sections: list[ReportSection]
    caveats: list[str] = Field(default_factory=list, description="Gaps or conflicts the reader must know about.")
    confidence: Confidence


class DirectAnswer(BaseModel):
    """Direct (single-agent RAG) output."""
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: Confidence
