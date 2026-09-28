from pathlib import Path
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', env_ignore_empty=True)
    openai_api_key: SecretStr = SecretStr('')
    chat_model: str = 'gpt-4.1-mini'
    embedding_model: str = 'text-embedding-3-small'
    chroma_path: Path = Path('storage/chroma')
    collection_name: str = 'knowledge_base'
    chunk_size: int = Field(1600, ge=200, le=2400)
    chunk_overlap: int = Field(200, ge=0)
    top_k: int = Field(5, ge=1, le=20)
    max_distance: float | None = Field(None, ge=0, le=2)
    context_max_chars: int = Field(12000, ge=2400, le=40000)
    max_output_tokens: int = Field(1200, ge=100, le=4000)
    app_api_key: SecretStr = SecretStr('')
    rag_api_url: str = 'http://127.0.0.1:8000'
    max_upload_bytes: int = 10 * 1024 * 1024

    @model_validator(mode='after')
    def check_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('CHUNK_OVERLAP must be smaller than CHUNK_SIZE')
        return self
