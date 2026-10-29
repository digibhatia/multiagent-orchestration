"""Look at what the RAG tool would return, WITHOUT any agent. Useful for tuning the
STRONG_MATCH / WEAK_MATCH distance thresholds in sutradhar/config.py.

    python inspect_kb.py "how many casual leave days do I get?"
"""
import sys

from sutradhar.kb import KnowledgeBase, format_results

q = " ".join(sys.argv[1:]) or "how many casual leave days do I get?"
print(format_results(KnowledgeBase().search(q)))
