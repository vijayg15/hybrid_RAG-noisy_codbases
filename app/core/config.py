from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", 
        env_file_encoding="utf-8", 
        extra="ignore", 
        case_sensitive=False,
    )

    app_env: str = "dev"
    # data_dir: Path = Path("./data")
    data_dir: Path = PROJECT_ROOT / "data"
    # database_url: str = "sqlite:///./data/rag.db"
    database_url: str = (
        f"sqlite:///{(PROJECT_ROOT / 'data' / 'rag.db').as_posix()}"
    )
    # qdrant_path: Path = Path("./data/qdrant")
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    openai_api_key: str = Field(min_length=1)
    openai_model: str = "gpt-4.1-mini"
    max_candidates: int = Field(default=40, ge=1, le=500)
    final_top_k: int = Field(default=8, ge=1, le=100)
    max_context_tokens: int = Field(default=12_000, ge=100)

    deepeval_model: str = "gpt-4.1-mini"
    deepeval_threshold: float = 0.7
    deepeval_include_reason: bool = True

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "repos").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
