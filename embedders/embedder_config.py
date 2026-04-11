from dataclasses import dataclass

@dataclass
class EmbedderConfig:
    provider: str = "bge"
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 64
    openai_api_key: str | None = None