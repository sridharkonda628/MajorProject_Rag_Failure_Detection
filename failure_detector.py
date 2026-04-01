"""
Lightweight Failure Detection Module.

Computes three reliability signals:
  - sr: Retrieval Similarity (cosine similarity between query and retrieved docs)
  - so: Lexical Overlap  (token-level Jaccard between answer and context)
  - sn: NLI Support       (entailment probability via cross-encoder)

Then derives a composite failure probability and confidence score.
"""

import re
import numpy as np
from sentence_transformers import CrossEncoder

import embeddings as emb
from config import ALPHA, BETA, GAMMA, NLI_MODEL

# ── NLI cross-encoder (lazy loaded) ─────────────────────────────────
_nli_model: CrossEncoder | None = None


def _get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        print(f"[FailureDetector] Loading NLI model: {NLI_MODEL}")
        _nli_model = CrossEncoder(NLI_MODEL)
        print("[FailureDetector] NLI model loaded.")
    return _nli_model


# ── Helper: tokenize ────────────────────────────────────────────────
def _tokenize(text: str) -> set[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return set(re.findall(r"\b\w+\b", text.lower()))


# ── Signal: Retrieval Similarity (sr) ───────────────────────────────
def compute_retrieval_similarity(query: str, doc_texts: list[str]) -> float:
    """
    Compute the mean cosine similarity between the query embedding
    and each retrieved document embedding.

    Returns a float in [0, 1].
    """
    if not doc_texts:
        return 0.0

    query_vec = emb.encode_single(query)
    doc_vecs = emb.encode(doc_texts)

    # cosine similarity (vectors are already L2-normalized)
    similarities = doc_vecs @ query_vec
    sr = float(np.mean(similarities))

    # Clamp to [0, 1]
    return max(0.0, min(1.0, sr))


# ── Signal: Lexical Overlap (so) ────────────────────────────────────
def compute_lexical_overlap(answer: str, context_docs: list[str]) -> float:
    """
    Compute token-level Jaccard overlap between the answer
    and the concatenated context documents.

    Returns a float in [0, 1].
    """
    if not answer or not context_docs:
        return 0.0

    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(" ".join(context_docs))

    if not answer_tokens or not context_tokens:
        return 0.0

    intersection = answer_tokens & context_tokens
    # Overlap is the percentage of answer tokens that appear in the context
    so = len(intersection) / len(answer_tokens)
    return max(0.0, min(1.0, so))


# ── Signal: NLI Support (sn) ────────────────────────────────────────
def compute_nli_support(answer: str, context_docs: list[str]) -> float:
    """
    Use a cross-encoder NLI model to compute the entailment probability
    between the context (premise) and the answer (hypothesis).

    The model outputs scores for [contradiction, neutral, entailment].
    We return the softmax probability of entailment.

    Returns a float in [0, 1].
    """
    if not answer or not context_docs:
        return 0.0

    model = _get_nli_model()

    max_sn = 0.0
    # Evaluate entailment against each document chunk independently
    # to avoid max_length truncation and return the highest support.
    for doc in context_docs:
        doc_text = doc[:2048]  # safe limit
        scores = model.predict([(doc_text, answer)])
        logits = scores[0] if len(scores.shape) > 1 else scores

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Entailment is class 1 for nli-deberta-v3-small
        sn = float(probs[1]) if len(probs) == 3 else float(probs[-1])
        if sn > max_sn:
            max_sn = sn

    return max(0.0, min(1.0, max_sn))


# ── Composite Score ─────────────────────────────────────────────────
def compute_failure_score(sr: float, so: float, sn: float) -> dict:
    """
    Compute failure probability and confidence from the three signals.

    P(failure) = 1 - (α·sr + β·so + γ·sn)
    Confidence = 1 - P(failure)

    Returns dict with all signals and derived scores.
    """
    p_failure = 1.0 - (ALPHA * sr + BETA * so + GAMMA * sn)
    p_failure = max(0.0, min(1.0, p_failure))
    confidence = 1.0 - p_failure

    return {
        "sr": round(sr, 4),
        "so": round(so, 4),
        "sn": round(sn, 4),
        "p_failure": round(p_failure, 4),
        "confidence": round(confidence, 4),
    }


# ── All-in-one convenience ──────────────────────────────────────────
def detect(
    query: str,
    answer: str,
    context_docs: list[str],
) -> dict:
    """
    Run full failure detection pipeline.

    Returns dict with sr, so, sn, p_failure, confidence.
    """
    sr = compute_retrieval_similarity(query, context_docs)
    so = compute_lexical_overlap(answer, context_docs)
    sn = compute_nli_support(answer, context_docs)

    return compute_failure_score(sr, so, sn)
