from __future__ import annotations
import math
from collections import defaultdict
import networkx as nx
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from app.core.config import get_settings
from app.domain.models import CodeChunk, RetrievedChunk
from app.storage.chunk_store import ChunkStore
from app.services.indexing import VectorIndex


class HybridRetriever:
    def __init__(self, store: ChunkStore, vector_index: VectorIndex) -> None:
        self.settings = get_settings()
        self.store = store
        self.vector_index = vector_index
        self.reranker = CrossEncoder(self.settings.reranker_model)

    def retrieve(self, query: str, repo_id: str | None = None, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.final_top_k
        all_chunks = self.store.get_chunks(repo_id)
        if not all_chunks:
            return []
        by_id = {c.chunk_id: c for c in all_chunks}

        dense = self.vector_index.search(query, self.settings.max_candidates, repo_id)
        sparse = self._bm25(query, all_chunks, self.settings.max_candidates)
        fused = self._rrf(dense, sparse)

        candidate_ids = [cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[: self.settings.max_candidates]]
        candidate_ids = self._dependency_expand(query, candidate_ids, all_chunks, limit=self.settings.max_candidates)
        candidates = [by_id[cid] for cid in candidate_ids if cid in by_id]

        pairs = [(query, f"{c.file_path}\n{c.symbol_name}\n{c.content}") for c in candidates]
        rerank_scores = self.reranker.predict(pairs).tolist() if pairs else []
        dense_map, sparse_map = dict(dense), dict(sparse)
        results = [
            RetrievedChunk(
                chunk=c,
                dense_score=dense_map.get(c.chunk_id, 0.0),
                sparse_score=sparse_map.get(c.chunk_id, 0.0),
                fusion_score=fused.get(c.chunk_id, 0.0),
                rerank_score=float(score),
            ) for c, score in zip(candidates, rerank_scores)
        ]
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.lower() for t in text.replace(".", " ").replace("(", " ").replace(")", " ").split() if t]

    def _bm25(self, query: str, chunks: list[CodeChunk], limit: int) -> list[tuple[str, float]]:
        corpus = [self._tokenize(f"{c.file_path} {c.symbol_name} {c.content}") for c in chunks]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(self._tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        max_score = max((float(scores[i]) for i in order), default=1.0) or 1.0
        return [(chunks[i].chunk_id, float(scores[i]) / max_score) for i in order if scores[i] > 0]

    @staticmethod
    def _rrf(dense: list[tuple[str, float]], sparse: list[tuple[str, float]], k: int = 60) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for ranking in (dense, sparse):
            for rank, (cid, _) in enumerate(ranking, start=1):
                scores[cid] += 1.0 / (k + rank)
        return dict(scores)

    def _dependency_expand(self, query: str, seed_ids: list[str], chunks: list[CodeChunk], limit: int) -> list[str]:
        by_symbol: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            by_symbol[c.symbol_name.lower()].append(c.chunk_id)
            by_symbol[c.symbol_name.split(".")[-1].lower()].append(c.chunk_id)
        graph = nx.DiGraph()
        for c in chunks:
            graph.add_node(c.chunk_id)
            for dep in c.dependencies:
                dep_key = dep.split(".")[-1].lower()
                for target in by_symbol.get(dep_key, []):
                    if target != c.chunk_id:
                        graph.add_edge(c.chunk_id, target)
        out = list(seed_ids)
        seen = set(out)
        q_terms = set(self._tokenize(query))
        for cid in seed_ids[:15]:
            if cid not in graph:
                continue
            neighbors = list(graph.successors(cid)) + list(graph.predecessors(cid))
            neighbors.sort(key=lambda nid: len(q_terms & set(self._tokenize(next((c.symbol_name for c in chunks if c.chunk_id == nid), "")))), reverse=True)
            for nid in neighbors:
                if nid not in seen:
                    out.append(nid)
                    seen.add(nid)
                    if len(out) >= limit:
                        return out
        return out
