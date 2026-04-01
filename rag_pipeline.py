"""
End-to-end RAG Pipeline with Lightweight Failure Detection.

Orchestrates: retrieve → generate → detect failure → attribute → control
              → (optional re-retrieve + regenerate) → final response.
"""

import logging
import time
from dataclasses import dataclass, field, asdict

from vector_store import get_store
from generator import generate, rewrite_query
from failure_detector import detect
from error_attribution import attribute
from adaptive_control import decide, ABSTAIN_MESSAGE
from config import TOP_K_INITIAL, TOP_K_RERETRIEVAL

# ── Logging ──────────────────────────────────────────────────────────
logger = logging.getLogger("rag_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


@dataclass
class PipelineResult:
    """Structured result from the RAG pipeline."""
    query: str = ""
    answer: str = ""
    sr: float = 0.0
    so: float = 0.0
    sn: float = 0.0
    confidence: float = 0.0
    p_failure: float = 0.0
    failure_type: str = ""
    decision: str = ""
    retrieved_docs: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    was_reretrieval: bool = False
    rewritten_query: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def run(query: str) -> PipelineResult:
    """
    Execute the full RAG + failure detection pipeline for a given query.

    Steps:
        1. Retrieve top-k documents
        2. Generate answer using LLM
        3. Run failure detection (sr, so, sn)
        4. Attribute failure type
        5. Apply adaptive control policy
        6. If re-retrieve: rewrite query, retrieve k=10, regenerate, re-detect
        7. Return structured result
    """
    start = time.time()
    result = PipelineResult(query=query)

    store = get_store()

    # ── Step 1: Initial Retrieval ────────────────────────────────────
    logger.info(f"Query: {query}")
    search_results = store.search(query, top_k=TOP_K_INITIAL)
    context_docs = [r["text"] for r in search_results]
    result.retrieved_docs = context_docs
    result.retrieval_scores = [r["score"] for r in search_results]

    if not context_docs:
        result.answer = ABSTAIN_MESSAGE
        result.decision = "abstain"
        result.failure_type = "Retrieval Failure"
        result.latency_ms = (time.time() - start) * 1000
        logger.warning("No documents retrieved. Abstaining.")
        return result

    # ── Step 2: Generate Answer ──────────────────────────────────────
    answer = generate(query, context_docs)
    result.answer = answer

    # ── Step 3: Failure Detection ────────────────────────────────────
    signals = detect(query, answer, context_docs)
    result.sr = signals["sr"]
    result.so = signals["so"]
    result.sn = signals["sn"]
    result.confidence = signals["confidence"]
    result.p_failure = signals["p_failure"]

    # ── Step 4: Error Attribution ────────────────────────────────────
    result.failure_type = attribute(result.sr, result.so, result.sn)

    # ── Step 5: Adaptive Control ─────────────────────────────────────
    decision = decide(result.confidence)
    result.decision = decision

    _log_diagnostics(result)

    # ── Step 6: Re-retrieval (if needed) ─────────────────────────────
    if decision == "re-retrieve":
        logger.info("Confidence moderate → re-retrieving with k=10 and rewritten query.")
        result.was_reretrieval = True

        # Rewrite query for better retrieval
        rewritten = rewrite_query(query)
        result.rewritten_query = rewritten
        logger.info(f"Rewritten query: {rewritten}")

        # Re-retrieve with higher k
        search_results = store.search(rewritten, top_k=TOP_K_RERETRIEVAL)
        context_docs = [r["text"] for r in search_results]
        result.retrieved_docs = context_docs
        result.retrieval_scores = [r["score"] for r in search_results]

        # Regenerate
        answer = generate(query, context_docs)
        result.answer = answer

        # Re-detect
        signals = detect(query, answer, context_docs)
        result.sr = signals["sr"]
        result.so = signals["so"]
        result.sn = signals["sn"]
        result.confidence = signals["confidence"]
        result.p_failure = signals["p_failure"]

        # Re-attribute and re-decide
        result.failure_type = attribute(result.sr, result.so, result.sn)
        decision = decide(result.confidence)
        result.decision = decision

        _log_diagnostics(result, prefix="[Re-retrieval]")

    # ── Step 7: Abstain if still low confidence ──────────────────────
    if decision == "abstain":
        result.answer = ABSTAIN_MESSAGE
        logger.info("Final decision: ABSTAIN")

    result.latency_ms = round((time.time() - start) * 1000, 2)
    logger.info(f"Pipeline completed in {result.latency_ms} ms")

    return result


def _log_diagnostics(result: PipelineResult, prefix: str = "") -> None:
    """Log diagnostic signals."""
    logger.info(
        f"{prefix} Diagnostics → "
        f"sr={result.sr:.4f}, so={result.so:.4f}, sn={result.sn:.4f}, "
        f"confidence={result.confidence:.4f}, "
        f"failure_type={result.failure_type}, decision={result.decision}"
    )
