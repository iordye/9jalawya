# src/config/settings.py

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    """
    LLM application settings.
    """
    model: str = "gemini-2.5-flash-lite"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, gt=0)
    google_api_key: str


class RetrieverSettings(BaseModel):
    """
    Retriever settings.
    """
    top_k: int = Field(default=3, gt=0)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class Settings(BaseSettings):
    """
    Root application settings.
    """

    llm: LLMSettings
    retriever: RetrieverSettings = RetrieverSettings()
    search_threshold: float = 0.5

    model_config = SettingsConfigDict(
        env_file="src\.env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )