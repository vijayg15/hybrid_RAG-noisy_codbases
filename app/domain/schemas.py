from __future__ import annotations
from pydantic import BaseModel, Field, HttpUrl


class IngestRequest(BaseModel):
    repo_url: str
    token: str | None = Field(default=None, repr=False)
    branch: str | None = None
    include_globs: list[str] = ["**/*"]
    exclude_globs: list[str] = [
        "**/.git/**", "**/node_modules/**", "**/.venv/**", "**/venv/**",
        "**/dist/**", "**/build/**", "**/__pycache__/**", "**/*.min.js",
        "**/vendor/**", "**/target/**",
    ]


class IngestResponse(BaseModel):
    repo_id: str
    commit_sha: str
    files_seen: int
    chunks_indexed: int
    embeddings_reused: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    repo_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    chunk_id: str
    symbol_name: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: int


class EvalRequest(BaseModel):
    dataset_path: str = "data/eval/sample_qrels.jsonl"
    repo_id: str | None = None
    k: int = Field(default=8, ge=1, le=50)


# class EvalItem(BaseModel):
#     question: str
#     precision_at_k: float
#     recall_at_k: float
#     retrieved_chunk_ids: list[str]
#     relevant_chunk_ids: list[str]


class DeepEvalMetricResult(BaseModel):
    score: float | None = None
    threshold: float
    success: bool
    reason: str | None = None
    error: str | None = None


class EvalItem(BaseModel):
    question: str

    actual_output: str
    expected_output: str | None = None

    precision_at_k: float
    recall_at_k: float

    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)

    contextual_precision: DeepEvalMetricResult | None = None
    contextual_recall: DeepEvalMetricResult | None = None
    faithfulness: DeepEvalMetricResult | None = None
    answer_relevancy: DeepEvalMetricResult | None = None


# class EvalResponse(BaseModel):
#     mean_context_precision: float
#     mean_context_recall: float
#     items: list[EvalItem]


class EvalResponse(BaseModel):
    total_items: int

    mean_context_precision: float
    mean_context_recall: float

    mean_deepeval_contextual_precision: float | None = None
    mean_deepeval_contextual_recall: float | None = None
    mean_faithfulness: float | None = None
    mean_answer_relevancy: float | None = None

    pass_rate: float | None = None

    items: list[EvalItem] = Field(default_factory=list)
