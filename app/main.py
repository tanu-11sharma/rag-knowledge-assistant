"""
FastAPI app exposing the RAG knowledge assistant.

Endpoints:
- GET  /health          liveness check
- GET  /docs-info        list of loaded documents
- POST /ask              ask a question, get a cited, grounded answer
"""

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import TfidfIndex, extractive_answer, load_corpus

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

app = FastAPI(
    title="RAG Knowledge Assistant",
    description=(
        "A small retrieval-augmented Q&A agent over a bundled sample "
        "document set. Answers are extractive and always cite the source "
        "chunk they came from."
    ),
    version="0.1.0",
)

_index = TfidfIndex()
_doc_count = load_corpus(_index, DOCS_DIR)


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    sentence: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents_loaded": _doc_count}


@app.get("/docs-info")
def docs_info() -> dict:
    doc_ids = sorted({c.doc_id for c in _index.chunks})
    return {"documents": doc_ids, "chunk_count": len(_index.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    retrieved = _index.search(req.question, top_k=req.top_k)
    result = extractive_answer(req.question, retrieved)
    return AskResponse(
        question=req.question,
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
    )
