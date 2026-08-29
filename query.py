"""Standalone CLI for testing the actual RAG chain used by app.py.

Previously this built its own separate chain (generic hub prompt, no coach
filter, no speaker-attribution guardrail, no metadata in context) — it had
drifted completely out of sync with app.py and would give different,
lower-quality answers than the real bot. Now it just calls the same
query_rag() the app itself uses.
"""
import sys
from app import query_rag

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many sets should I do for each exercise?"

    result = query_rag(question)

    print(f"Q: {question}\n")
    print(f"A: {result['answer']}\n")
    print(f"--- {len(result['context'])} chunks retrieved ---")
    for i, chunk_text in enumerate(result["context"], 1):
        print(f"\n[{i}] {chunk_text[:200]}...")
