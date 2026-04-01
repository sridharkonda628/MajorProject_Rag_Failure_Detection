"""
LLM Generator module.
Uses Google Gemini API to generate answers from query + context.
"""

import time
from google import genai
from config import GEMINI_API_KEY, LLM_MODEL, USE_MOCK_LLM

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Initialize and cache the Gemini client."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Please set it in a .env file or as an environment variable."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def generate(query: str, context_docs: list[str]) -> str:
    """
    Generate an answer using Gemini given a query and retrieved context documents.

    Args:
        query: The user's question.
        context_docs: List of retrieved document texts used as context.

    Returns:
        The generated answer string.
    """
    context_text = "\n\n---\n\n".join(context_docs)

    if USE_MOCK_LLM:
        time.sleep(1.5)  # Simulate network latency
        if not context_docs:
            return "I cannot provide an answer because no relevant context was found."
        
        # EXPERIMENTAL OVERRIDE FOR PRESENTATION:
        if "hallucinate" in query.lower():
            return "The capital of France is Mars, and quantum computing is made of cheese."

        # Use exact sentences from context to ensure high NLI and overlap scores for demo
        first_doc = context_docs[0]
        sentences = [s.strip() + "." for s in first_doc.replace("\n", " ").split(".") if s.strip()]
        mock_answer = " ".join(sentences[:3]) if sentences else first_doc[:300]
        return f"[MOCK] {mock_answer}"

    prompt = (
        "You are a helpful and accurate question-answering assistant. "
        "Answer the user's question based ONLY on the provided context documents. "
        "If the context does not contain enough information to answer the question, "
        "say so clearly. Do not make up information.\n\n"
        f"Context Documents:\n\n{context_text}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    import logging
    logger = logging.getLogger("generator")

    # Retry loop with exponential backoff to handle 429 Too Many Requests
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = _get_client()
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1 and "429" in str(e):
                wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                logger.warning(f"[Generator] Rate limit hit (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            elif attempt < max_retries - 1 and "retryDelay" in str(e):
                wait_time = 5
                logger.warning(f"[Generator] Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"[Generator Error] {e}")
                return f"[Generator Error] {str(e)}"
    
    return "[Generator Error] Max retries exceeded."


def rewrite_query(original_query: str) -> str:
    """
    Use Gemini to rewrite a query for better retrieval.

    Args:
        original_query: The original user query.

    Returns:
        A rewritten version of the query.
    """
    prompt = (
        "You are a search query optimizer. Rewrite the following query "
        "to be more specific and descriptive for better document retrieval. "
        "Return only the rewritten query, nothing else.\n\n"
        f"Query: {original_query}"
    )

    import logging
    logger = logging.getLogger("generator")

    if USE_MOCK_LLM:
        time.sleep(0.5)
        return f"{original_query} detailed explanation and concepts"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = _get_client()
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1 and ("429" in str(e) or "retryDelay" in str(e)):
                wait_time = (2 ** attempt) * 2
                logger.warning(f"[Generator / Rewrite] Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"[Generator Error / Rewrite] {e}")
                return original_query

    return original_query
