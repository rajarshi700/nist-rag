from fastembed import TextEmbedding
from tokenizers import Tokenizer

from .config import AppError, Settings


class Embeddings:
    def __init__(self, settings: Settings):
        try:
            self.model = TextEmbedding(
                model_name=settings.embedding_model,
                cache_dir=str(settings.data_dir / "models"),
                threads=2,
            )
            # Clone it: turning off truncation here must not change inference settings.
            self.tokenizer = Tokenizer.from_str(self.model.model.tokenizer.to_str())
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
        except Exception as exc:
            raise AppError(
                "Could not load the embedding model. The first run needs internet access "
                "to Hugging Face. Check the connection and EMBEDDING_MODEL, then retry."
            ) from exc

    def documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=32)]

    def query(self, text: str) -> list[float]:
        return next(self.model.query_embed(text)).tolist()
