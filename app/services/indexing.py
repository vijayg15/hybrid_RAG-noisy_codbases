from __future__ import annotations
import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings
from app.domain.models import CodeChunk


class VectorIndex:
    collection = "code_chunks"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = QdrantClient(path=str(self.settings.qdrant_path))
        self.model = SentenceTransformer(self.settings.embedding_model)
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(
                "Could not determine embedding dimension."
            )
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection, 
                vectors_config=VectorParams(size=dim, 
                                            distance=Distance.COSINE,
                ),
            )

    def index(self, chunks: list[CodeChunk]) -> int:
        if not chunks:
            return 0
        texts = [self._embedding_text(c) for c in chunks]
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
        points = [
            PointStruct(
                id=self._numeric_id(c.chunk_id),
                vector=v.tolist(),
                payload={"chunk_id": c.chunk_id, "repo_id": c.repo_id, "content_hash": c.content_hash},
            ) for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(
            collection_name=self.collection, 
            points=points, 
            wait=True,
        )
        return 0  # Hook for an external embedding cache; IDs/content hashes already make re-ingestion idempotent.

    def search(self, query: str, limit: int, repo_id: str | None = None) -> list[tuple[str, float]]:
        vector = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        query_filter = None
        if repo_id:
            query_filter = Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))])
        # hits = self.client.search(collection_name=self.collection, query_vector=vector, query_filter=query_filter, limit=limit)
        response =self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        # return [(str(h.payload["chunk_id"]), float(h.score)) for h in hits]
        return [
            (
                str(hit.payload["chunk_id"]),
                float(hit.score),
            )
            for hit in response.points
            if hit.payload and "chunk_id" in hit.payload
        ]
    
    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _embedding_text(c: CodeChunk) -> str:
        return f"path: {c.file_path}\nsymbol: {c.symbol_name}\ntype: {c.symbol_type}\n{c.content}"

    @staticmethod
    def _numeric_id(chunk_id: str) -> int:
        return int(chunk_id[:15], 16)
