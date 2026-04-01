"""
Scenario demonstration script for the RAG Failure Detection System.
Run this script to see all three decision branches (Finalize, Re-retrieve, Abstain)
act out perfectly for demonstration purposes.
"""
import requests
import time
import json

API_URL = "http://127.0.0.1:8000"

# Note: The server must be running!
def wait_for_server():
    print("Checking if API is running...")
    for _ in range(5):
        try:
            requests.get(f"{API_URL}/health", timeout=2)
            print("API is UP!\n")
            return True
        except:
            time.sleep(1)
    print("API is down. Please run 'python main.py' first.")
    return False

def print_result(scenario, query, data):
    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {scenario}")
    print(f"QUERY:    \"{query}\"")
    print(f"{'-' * 70}")
    
    if "failure_type" not in data:
        print(f"ERROR: {data}")
        return

    # Print Signals
    print(f"Diagnostic Signals:")
    print(f"  • Retrieval Similarity (sr): {data['sr']:.4f}")
    print(f"  • Lexical Overlap      (so): {data['so']:.4f}")
    print(f"  • NLI Support          (sn): {data['sn']:.4f}")
    print(f"")
    print(f"Composite Scores:")
    print(f"  • P(failure): {data['p_failure']:.4f}")
    print(f"  • Confidence: {data['confidence']:.4f}")
    print(f"")
    print(f"System Decision:")
    print(f"  ▶ Failure Type: {data['failure_type']}")
    print(f"  ▶ Action Taken: {data['decision'].upper()}")
    if data['was_reretrieval']:
        print(f"  ▶ Triggered Re-retrieval with rewrite: '{data['rewritten_query']}'")
    
    print(f"")
    print(f"Final Answer:")
    # Wrap text nicely
    import textwrap
    wrapped = textwrap.fill(data['answer'], width=68)
    for line in wrapped.split('\n'):
        print(f"  {line}")
    print(f"{'=' * 70}\n")

def run_scenarios():
    if not wait_for_server():
        return

    test_cases = [
        # SCENARIO 1: Success -> Finalize
        # Topic is exactly in the documents, clear answer.
        ("SUCCESS (Finalize)", "What is quantum computing and what are qubits?"),
        
        # SCENARIO 2: Retrieval Failure -> Abstain
        # Topic not in database, sr is low.
        ("RETRIEVAL FAILURE (Abstain)", "How do I bake a chocolate cake?"),
        
        # SCENARIO 3: Generation Failure -> Re-retrieve
        # Topic is retrieved perfectly (sr > 0.35), but the LLM hallucinated
        # the response completely, dropping overlap and NLI score below thresholds
        ("GENERATION FAILURE (Re-retrieve)", "hallucinate quantum computing.")
    ]

    for scenario, query in test_cases:
        try:
            print(f"Sending query for {scenario}...")
            # We bypass the mock if needed to show realistic numbers or rely on threshold tweaks
            r = requests.post(
                f"{API_URL}/ask",
                json={"query": query},
                timeout=120,
            )
            print_result(scenario, query, r.json())
            time.sleep(1) # brief pause for dramatic effect
        except Exception as e:
            print(f"Failed to execute scenario '{scenario}': {e}")

if __name__ == "__main__":
    run_scenarios()
