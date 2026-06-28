from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import create_engine, String, Integer, Text, DateTime, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


class RepositoryRecord(Base):
    __tablename__ = "repositories"
    repo_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    repo_url: Mapped[str] = mapped_column(Text)
    commit_sha: Mapped[str] = mapped_column(String(64))
    local_path: Mapped[str] = mapped_column(Text)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ChunkRecord(Base):
    __tablename__ = "chunks"
    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    repo_id: Mapped[str] = mapped_column(String(64), index=True)
    file_path: Mapped[str] = mapped_column(Text, index=True)
    language: Mapped[str] = mapped_column(String(32))
    symbol_name: Mapped[str] = mapped_column(Text, index=True)
    symbol_type: Mapped[str] = mapped_column(String(32))
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    dependencies_json: Mapped[str] = mapped_column(Text, default="[]")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    __table_args__ = (UniqueConstraint("repo_id", "file_path", "start_line", "end_line", name="uq_chunk_location"),)


settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
