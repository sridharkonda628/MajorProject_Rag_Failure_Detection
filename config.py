"""
Central configuration for the RAG Failure Detection System.
All thresholds, model names, and hyperparameters live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Model Configuration ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "False").lower() in ("true", "1", "t")

# ── Retrieval Configuration ──────────────────────────────────────────
TOP_K_INITIAL = 5
TOP_K_RERETRIEVAL = 10

# ── Failure Detection Weights ────────────────────────────────────────
ALPHA = 0.3   # weight for retrieval similarity (sr)
BETA = 0.2    # weight for lexical overlap (so)
GAMMA = 0.5   # weight for NLI support (sn)

# ── Attribution Thresholds ───────────────────────────────────────────
TAU_SIM = 0.35    # retrieval similarity threshold
TAU_NLI = 0.50    # NLI support threshold
TAU_OVER = 0.40   # lexical overlap threshold

# ── Adaptive Control Thresholds ──────────────────────────────────────
CONFIDENCE_HIGH = 0.7   # above → finalize answer
CONFIDENCE_LOW = 0.5    # at or below → abstain

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")
FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")

# ── API Configuration ───────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
