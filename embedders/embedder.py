import logging
import numpy as np
from embedders.embedder_config import EmbedderConfig

class Embedder:
    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        if cfg.provider == "bge":
            from sentence_transformers import SentenceTransformer
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            self._model = SentenceTransformer(cfg.model_name, trust_remote_code=True)
        elif cfg.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=cfg.openai_api_key)
        else:
            raise ValueError(f"Unknown provider: {cfg.provider!r}. Use 'bge' or 'openai'.")

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.cfg.provider == "bge":
            return self._encode_bge(texts)
        return self._encode_openai(texts)

    def _encode_bge(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=self.cfg.batch_size,
        )
        return np.array(vecs, dtype=np.float32)

    def _encode_openai(self, texts: list[str]) -> np.ndarray:
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = texts[i: i + self.cfg.batch_size]
            response = self._client.embeddings.create(
                model=self.cfg.model_name, input=batch
            )
            all_vecs.extend(item.embedding for item in response.data)
            logging.info(f"Embedded {min(i + self.cfg.batch_size, len(texts))}/{len(texts)}")
        vecs = np.array(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)