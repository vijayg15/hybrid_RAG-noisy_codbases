from __future__ import annotations
from app.domain.schemas import IngestRequest, IngestResponse, QueryResponse
from app.services.repository import RepositoryService
from app.services.chunking import ASTChunker
from app.services.indexing import VectorIndex
from app.services.retrieval import HybridRetriever
from app.services.generation import AnswerGenerator
from app.storage.chunk_store import ChunkStore


class RAGPipeline:
    def __init__(self) -> None:
        self.repos = RepositoryService()
        self.chunker = ASTChunker()
        self.store = ChunkStore()
        self.vectors = VectorIndex()
        self.retriever = HybridRetriever(self.store, self.vectors)
        self.generator = AnswerGenerator()

    def ingest(self, req: IngestRequest) -> IngestResponse:
        repo_id, path, commit_sha = self.repos.clone_or_update(req.repo_url, req.token, req.branch)
        chunks, files_seen = self.chunker.chunk_repository(repo_id, path, req.include_globs, req.exclude_globs)
        self.store.upsert_repository(repo_id, req.repo_url, commit_sha, str(path))
        self.store.replace_repo_chunks(repo_id, chunks)
        reused = self.vectors.index(chunks)
        return IngestResponse(repo_id=repo_id, commit_sha=commit_sha, files_seen=files_seen, chunks_indexed=len(chunks), embeddings_reused=reused)

    def query(self, question: str, repo_id: str | None = None, top_k: int | None = None) -> QueryResponse:
        results = self.retriever.retrieve(question, repo_id, top_k)
        answer, citations = self.generator.generate(question, results)
        return QueryResponse(answer=answer, citations=citations, retrieved_chunks=len(results))
