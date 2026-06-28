from __future__ import annotations
import json
from pathlib import Path
from app.domain.schemas import EvalItem, EvalResponse
from app.services.retrieval import HybridRetriever


class EvaluationService:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def evaluate(self, dataset_path: str, repo_id: str | None, k: int) -> EvalResponse:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        items: list[EvalItem] = []
        for row in rows:
            relevant = set(row.get("relevant_chunk_ids", []))
            retrieved = self.retriever.retrieve(row["question"], repo_id=repo_id or row.get("repo_id"), top_k=k)
            ids = [r.chunk.chunk_id for r in retrieved]
            hits = len(set(ids) & relevant)
            precision = hits / len(ids) if ids else 0.0
            recall = hits / len(relevant) if relevant else 0.0
            items.append(EvalItem(question=row["question"], precision_at_k=precision, recall_at_k=recall, retrieved_chunk_ids=ids, relevant_chunk_ids=sorted(relevant)))
        n = len(items) or 1
        return EvalResponse(
            mean_context_precision=sum(i.precision_at_k for i in items) / n,
            mean_context_recall=sum(i.recall_at_k for i in items) / n,
            items=items,
        )
