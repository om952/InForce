from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    llm_provider: Literal["gemini"] = "gemini"
    gemini_api_key: SecretStr | None = None
    classifier_model: str = "gemini-3.5-flash-lite"
    llm_timeout_seconds: float = 30

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )
    embedding_cache_dir: str = "data/models"
    lexical_model: str = "Qdrant/bm25"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    qdrant_url: str | None = None
    qdrant_path: str = "data/qdrant"
    qdrant_collection: str = "sebi_mf_chunks"

    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("inforce-local")
    neo4j_database: str = "neo4j"
