"""
Adaptive Control Policy.

Decides the system action based on confidence score:
  - C > 0.8          → Finalize answer
  - 0.4 < C ≤ 0.8    → Re-retrieve (k=10) and regenerate
  - C ≤ 0.4          → Abstain
"""

from config import CONFIDENCE_HIGH, CONFIDENCE_LOW

ABSTAIN_MESSAGE = (
    "I cannot provide a reliable answer based on the available documents."
)


def decide(confidence: float) -> str:
    """
    Decide the control action based on the confidence score.

    Args:
        confidence: Composite confidence score [0, 1].

    Returns:
        One of: "finalize", "re-retrieve", "abstain".
    """
    if confidence > CONFIDENCE_HIGH:
        return "finalize"
    elif confidence > CONFIDENCE_LOW:
        return "re-retrieve"
    else:
        return "abstain"
