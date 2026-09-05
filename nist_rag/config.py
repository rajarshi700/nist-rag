import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class AppError(Exception):
    """An error we can explain without a traceback in the CLI."""


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    groq_model: str = "openai/gpt-oss-20b"
    api_key: str = ""
    top_k: int = 6
    timeout: float = 45
    max_api_retries: int = 2
    max_retry_wait: float = 30
    chunk_tokens: int = 220
    chunk_overlap: int = 32

    @classmethod
    def from_env(cls):
        load_dotenv(Path.cwd() / ".env")
        try:
            settings = cls(
                data_dir=Path(os.getenv("DATA_DIR", "./data")).expanduser(),
                embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
                groq_model=os.getenv("GROQ_MODEL", cls.groq_model),
                api_key=os.getenv("GROQ_API_KEY", "").strip(),
                top_k=int(os.getenv("TOP_K", "6")),
                timeout=float(os.getenv("HTTP_TIMEOUT", "45")),
                max_api_retries=int(os.getenv("MAX_API_RETRIES", "2")),
                max_retry_wait=float(os.getenv("MAX_RETRY_WAIT", "30")),
            )
        except ValueError as exc:
            raise AppError("Check the numeric settings in .env.") from exc
        if not 1 <= settings.top_k <= 10:
            raise AppError("TOP_K must be between 1 and 10.")
        if settings.timeout <= 0 or not 0 <= settings.max_api_retries <= 3:
            raise AppError("HTTP_TIMEOUT must be positive; MAX_API_RETRIES must be 0 to 3.")
        if not 0 <= settings.max_retry_wait <= 60:
            raise AppError("MAX_RETRY_WAIT must be between 0 and 60 seconds.")
        return settings

    @property
    def index_dir(self):
        return self.data_dir / "index"
