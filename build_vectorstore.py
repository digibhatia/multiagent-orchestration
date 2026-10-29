"""STEP 1 - build the knowledge base (run once, and again whenever documents change).

    python build_vectorstore.py
"""
from sutradhar.kb import KnowledgeBase

n = KnowledgeBase().build()
print(f"Done. Stored {n} chunks from knowledge_base/ into chroma_db/")
print("Try:  python inspect_kb.py \"how many casual leave days do I get?\"")
