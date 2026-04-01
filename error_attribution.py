"""
Error Attribution Module.

Determines the source of failure based on diagnostic signals:
  - Retrieval Failure:   sr < TAU_SIM
  - Generation Failure:  sr >= TAU_SIM AND (sn < TAU_NLI OR so < TAU_OVER)
  - Success:             otherwise
"""

from config import TAU_SIM, TAU_NLI, TAU_OVER


def attribute(sr: float, so: float, sn: float) -> str:
    """
    Determine the error attribution type.

    Args:
        sr: Retrieval similarity score [0, 1].
        so: Lexical overlap score [0, 1].
        sn: NLI support score [0, 1].

    Returns:
        One of: "Retrieval Failure", "Generation Failure", "Success".
    """
    if sr < TAU_SIM:
        return "Retrieval Failure"
    elif sn < TAU_NLI or so < TAU_OVER:
        return "Generation Failure"
    else:
        return "Success"
