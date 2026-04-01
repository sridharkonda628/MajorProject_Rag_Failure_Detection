"""Full end-to-end test script for the RAG Failure Detection System."""
import requests
import json

API_URL = "http://127.0.0.1:8000"

test_queries = [
    ("What is quantum computing?", "Should succeed — in-domain"),
    ("How does photosynthesis work?", "Should succeed — in-domain"),
    ("How to cook pasta?", "Should abstain — out-of-domain"),
]

print("=" * 70)
print("RAG FAILURE DETECTION SYSTEM — END-TO-END TEST")
print("=" * 70)

# Health check
r = requests.get(f"{API_URL}/health", timeout=10)
print(f"\n[Health Check] Status: {r.status_code} | {r.json()}")

for query, description in test_queries:
    print(f"\n{'─' * 70}")
    print(f"TEST: {description}")
    print(f"Query: \"{query}\"")
    print(f"{'─' * 70}")

    try:
        r = requests.post(
            f"{API_URL}/ask",
            json={"query": query},
            timeout=120,
        )
        data = r.json()

        print(f"  Answer:       {data['answer'][:200]}")
        print(f"  Confidence:   {data['confidence']}")
        print(f"  sr (retrieval): {data['sr']}")
        print(f"  so (overlap):   {data['so']}")
        print(f"  sn (NLI):       {data['sn']}")
        print(f"  P(failure):   {data['p_failure']}")
        print(f"  Failure Type: {data['failure_type']}")
        print(f"  Decision:     {data['decision']}")
        print(f"  Re-retrieval: {data['was_reretrieval']}")
        print(f"  Latency:      {data['latency_ms']} ms")
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'=' * 70}")
print("TEST COMPLETE")
print("=" * 70)
