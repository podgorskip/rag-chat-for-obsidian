from typing import Any
from torch import Tensor
from rags.llm_client import LLMClient
from rags.rag_config import Config
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
import pandas as pd

class RAG:
  def __init__(
        self,
        client: LLMClient,
        embedding_model: SentenceTransformer,
        df: pd.DataFrame,
        config: Config = Config(),
        llm_model: str='llama3'
      ):

    self.client = client
    self.embedding_model = embedding_model
    self.df = df.copy()
    self.config = config
    self.llm_model = llm_model

    self.cumulative_tokens = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
        }

  def _prepare_chunks(self, candidates: pd.DataFrame) -> list[dict[str, int | Any]]:
    chunks = []
    prev_sim = candidates.iloc[0]['similarity']

    for row in candidates.itertuples(index=False):
      curr_sim = row.similarity
      delta = abs(curr_sim - prev_sim)

      if delta > self.config.DELTA_CUTOFF:
        break

      chunks.append({
          "title": row.title,
          "content": row.content,
          "similarity": curr_sim,
          "length": len(row.content)
      })
      prev_sim = curr_sim

    return chunks

  def _track_tokens(self, usage) -> None:
      if self.client.provider != "openai":
          return
      if usage:
          self.cumulative_tokens["prompt_tokens"] += usage.prompt_tokens
          self.cumulative_tokens["completion_tokens"] += usage.completion_tokens
          self.cumulative_tokens["total_tokens"] += usage.total_tokens

  def _estimate_tokens(self, text: str) -> int:
    return len(text) // 4

  def embed_query(self, query: str) -> Tensor:
    return self.embedding_model.encode(query, normalize_embeddings=True)

  def retrieve(self, query_embedding: Tensor) -> list:
    self.df['similarity'] = np.dot(np.stack(self.df['embedding']), query_embedding)
    threshold = self.df['similarity'].quantile(1 - self.config.TOP_K)
    candidates = self.df[self.df['similarity'] >= threshold]
    candidates = candidates[candidates['similarity'] > self.config.MIN_SIMILARITY]

    if candidates.empty:
      return []

    return self._prepare_chunks(candidates)

  def rephrase_query(self, original_query: str, history: list = None) -> str:
      messages = [{"role": "system", "content": self.config.REPHRASE_PROMPT}]

      if history:
          messages.extend(history[-4:])

      messages.append({"role": "user", "content": original_query})

      response = self.client.chat.completions.create(
          model=self.llm_model,
          messages=messages
      )
      self._track_tokens(response.usage)
      return response.choices[0].message.content.strip()

  def build_context(self, chunks: list) -> str:
    context_text = ""
    current_context_tokens = 0

    for chunk in chunks:
      chunk_text = f"PASSAGE ({chunk['title']}): '{chunk['content']}'\n"
      chunk_tokens = self._estimate_tokens(chunk_text)

      if current_context_tokens + chunk_tokens <= self.config.MAX_CONTEXT_TOKENS:
        context_text += chunk_text
        current_context_tokens += chunk_tokens
      else:
        logging.info(f"Context tokens exceeded (~{current_context_tokens}).")
        break

    return context_text

  def ask(self, query: str):
    query_embedding = self.embed_query(query)
    chunks = self.retrieve(query_embedding)

    if not chunks:
      rephrased = self.rephrase_query(query)
      rephrased_embedding = self.embed_query(rephrased)
      chunks = self.retrieve(rephrased_embedding)

    if not chunks:
      return None

    context = self.build_context(chunks)
    prompt = f"QUESTION: '{query}'\n\n{context}\nANSWER:"

    response = self.client.chat.completions.create(
        model=self.llm_model,
        messages=[
            {"role": "system", "content": self.config.ANSWER_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    self._track_tokens(response.usage)
    return response.choices[0].message.content, response.usage, self.cumulative_tokens