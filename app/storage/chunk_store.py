from __future__ import annotations
import json
from sqlalchemy import delete, select
from app.domain.models import CodeChunk
from app.storage.database import ChunkRecord, RepositoryRecord, SessionLocal


class ChunkStore:
    def upsert_repository(self, repo_id: str, repo_url: str, commit_sha: str, local_path: str) -> None:
        with SessionLocal() as session:
            row = session.get(RepositoryRecord, repo_id)
            if row is None:
                row = RepositoryRecord(repo_id=repo_id, repo_url=repo_url, commit_sha=commit_sha, local_path=local_path)
                session.add(row)
            else:
                row.repo_url, row.commit_sha, row.local_path = repo_url, commit_sha, local_path
            session.commit()

    def replace_repo_chunks(self, repo_id: str, chunks: list[CodeChunk]) -> None:
        with SessionLocal() as session:
            session.execute(delete(ChunkRecord).where(ChunkRecord.repo_id == repo_id))
            session.add_all([
                ChunkRecord(
                    chunk_id=c.chunk_id, repo_id=c.repo_id, file_path=c.file_path,
                    language=c.language, symbol_name=c.symbol_name, symbol_type=c.symbol_type,
                    start_line=c.start_line, end_line=c.end_line, content=c.content,
                    content_hash=c.content_hash, dependencies_json=json.dumps(c.dependencies),
                    metadata_json=json.dumps(c.metadata),
                ) for c in chunks
            ])
            session.commit()

    def get_chunks(self, repo_id: str | None = None) -> list[CodeChunk]:
        with SessionLocal() as session:
            stmt = select(ChunkRecord)
            if repo_id:
                stmt = stmt.where(ChunkRecord.repo_id == repo_id)
            rows = session.scalars(stmt).all()
        return [self._to_domain(r) for r in rows]

    def get_by_ids(self, ids: list[str]) -> list[CodeChunk]:
        if not ids:
            return []
        with SessionLocal() as session:
            rows = session.scalars(select(ChunkRecord).where(ChunkRecord.chunk_id.in_(ids))).all()
        order = {cid: i for i, cid in enumerate(ids)}
        return sorted((self._to_domain(r) for r in rows), key=lambda c: order.get(c.chunk_id, 10**9))

    @staticmethod
    def _to_domain(r: ChunkRecord) -> CodeChunk:
        return CodeChunk(
            chunk_id=r.chunk_id, repo_id=r.repo_id, file_path=r.file_path,
            language=r.language, symbol_name=r.symbol_name, symbol_type=r.symbol_type,
            start_line=r.start_line, end_line=r.end_line, content=r.content,
            content_hash=r.content_hash, dependencies=json.loads(r.dependencies_json or "[]"),
            metadata=json.loads(r.metadata_json or "{}"),
        )
