from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

# qdrant_url: str
# qdrant_api_key: str
# docs_collection: str = "docs"
# interactions_collection: str = "interactions"
# chunk_size: int = 1000
# chunk_overlap: int = 200
# embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class Settings(BaseSettings):
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_model: str = Field("llama3-70b-8192", alias="GROQ_MODEL")

    openweather_api_key: str = Field(..., alias="OPENWEATHER_API_KEY")

    qdrant_url: str = Field("http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")

    # LangSmith / LangChain tracing
    langsmith_api_key: Optional[str] = Field(
        default=None, alias="LANGSMITH_API_KEY")
    langchain_tracing: Optional[bool] = Field(
        default=True, alias="LANGCHAIN_TRACING_V2")
    langchain_project: Optional[str] = Field(
        default="weather-rag-groq-demo", alias="LANGCHAIN_PROJECT")

    # Embeddings / vectorstore
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    docs_collection: str = Field("docs", alias="DOCS_COLLECTION")
    interactions_collection: str = Field(
        "interactions", alias="INTERACTIONS_COLLECTION")

    chunk_size: int = Field(1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(120, alias="CHUNK_OVERLAP")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
