from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.domain.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, EvalRequest, EvalResponse
from app.services.evaluation import EvaluationService
from app.services.pipeline import RAGPipeline

router = APIRouter()
pipeline = RAGPipeline()
evaluator = EvaluationService(pipeline.retriever)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    try:
        return pipeline.ingest(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        return pipeline.query(req.question, req.repo_id, req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/eval", response_model=EvalResponse)
def evaluate(req: EvalRequest) -> EvalResponse:
    try:
        return evaluator.evaluate(req.dataset_path, req.repo_id, req.k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/eval", response_model=EvalResponse)
def evaluate_default() -> EvalResponse:
    return evaluator.evaluate("data/eval/sample_qrels.jsonl", None, 8)
