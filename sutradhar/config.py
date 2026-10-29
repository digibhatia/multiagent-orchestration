"""Every tunable number in ONE place. Change it here, restart, and watch behaviour change."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "knowledge_base"
CHROMA_DIR = str(ROOT / "chroma_db")

# --- Models -----------------------------------------------------------------
# Two knobs on purpose: the ROUTER sees every question, so it should be cheap and fast.
# The WORKERS (retrieval, analyst, writer, direct) do the real thinking.
# (Curriculum model is GPT-4o; set it here if your account still serves it.)
MODEL_ROUTER = os.getenv("MODEL_ROUTER", "gpt-5.4-mini")
MODEL_WORKERS = os.getenv("MODEL_WORKERS", "gpt-5.4-mini")
EMBED_MODEL = "text-embedding-3-small"

# --- Retrieval (same meaning as Day 9) -----------------------------------------
TOP_K = 4
STRONG_MATCH = 0.55      # cosine distance: lower = closer
WEAK_MATCH = 0.75        # best hit above this => treat as "nothing relevant found"
MAX_EVIDENCE = 6         # pipeline keeps at most this many chunks for the analyst
MAX_SEARCHES = 3         # the retrieval agent may search at most this many times

# --- Routing ----------------------------------------------------------------
DEFAULT_ROUTE_MODE = "auto"   # auto | always_direct | always_pipeline
SHORT_QUESTION_WORDS = 12     # rule: this short + no 'and' => direct
LONG_QUESTION_WORDS = 25      # rule: longer than this => pipeline

# --- Human-in-the-loop ---------------------------------------------------------
REQUIRE_APPROVAL = os.getenv("REQUIRE_APPROVAL", "true").lower() == "true"
HIGH_RISK_TOPICS = ["terminate", "termination", "disciplinary", "harassment", "dismiss",
                    "fired", "legal", "lawsuit", "misconduct", "breach", "stolen", "theft"]

# --- Writer validation loop -------------------------------------------------------
MAX_WRITER_ATTEMPTS = 2       # first try + one retry with feedback, then fall back

# --- Safety loop guard (per agent run) ------------------------------------------
AGENT_REQUEST_LIMIT = 8

REFUSAL_TEXT = ("I could not find this in the approved company documents, so I will not guess. "
                "Please contact HR or the IT helpdesk.")
FALLBACK_TEXT = ("I could not produce an answer that passed the citation checks, so I am not showing a draft. "
                 "Please rephrase the question or contact HR / the IT helpdesk.")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
