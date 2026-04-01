"""
FastAPI backend for the RAG Failure Detection System.
Exposes /ask endpoint and a /health check.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_pipeline

app = FastAPI(
    title="RAG Failure Detection System",
    description="Lightweight Diagnostic Failure Detection and Adaptive Control for Reliable RAG Systems",
    version="1.0.0",
)

# CORS – allow Streamlit and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sr: float
    so: float
    sn: float
    confidence: float
    p_failure: float
    failure_type: str
    decision: str
    retrieved_docs: list[str]
    retrieval_scores: list[float]
    was_reretrieval: bool
    rewritten_query: str
    latency_ms: float


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    Process a user query through the RAG + failure detection pipeline.
    Returns the answer along with all diagnostic signals.
    """
    result = rag_pipeline.run(request.query)
    return QueryResponse(**result.to_dict())


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT

    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
