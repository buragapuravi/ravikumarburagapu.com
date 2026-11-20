from __future__ import annotations

import os

from app.config import settings


def main() -> None:
    config = settings()
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from sentence_transformers import SentenceTransformer
    print(f"Downloading/warming embedding model: {config.embedding_model}")
    SentenceTransformer(config.embedding_model, device=config.embedding_device)
    if config.reranker_model:
        from sentence_transformers import CrossEncoder
        print(f"Downloading/warming reranker: {config.reranker_model}")
        CrossEncoder(config.reranker_model)
    print("Model cache is ready. The app is configured for offline model loading.")


if __name__ == "__main__":
    main()
