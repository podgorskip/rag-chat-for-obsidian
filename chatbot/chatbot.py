from typing import Callable
from rags.rag import RAG

class Chatbot:
    def __init__(self, rag: RAG):
        self.rag = rag
        self.conversation_history: list[dict] = []

    def _track_tokens(self, usage) -> None:
        if self.rag.client.provider != "openai":
            return
        if usage:
            self.rag.cumulative_tokens["prompt_tokens"] += usage.prompt_tokens
            self.rag.cumulative_tokens["completion_tokens"] += usage.completion_tokens
            self.rag.cumulative_tokens["total_tokens"] += usage.total_tokens

    def chat(
        self,
        query: str,
        confirm_external: Callable[[], bool] | None = None,
    ):
        query_embedding = self.rag.embed_query(query)
        chunks = self.rag.retrieve(query_embedding)

        if not chunks:
            rephrased = self.rag.rephrase_query(query, history=self.conversation_history)
            rephrased_embedding = self.rag.embed_query(rephrased)
            chunks = self.rag.retrieve(rephrased_embedding)

        context = self.rag.build_context(chunks) if chunks else ""

        if not context:
            if confirm_external is not None and not confirm_external():
                return None

        user_turn = f"QUESTION: '{query}'\n\n{context}\nANSWER:" if context else query
        self.conversation_history.append({"role": "user", "content": user_turn})

        messages = [
            {"role": "system", "content": self.rag.config.ANSWER_PROMPT},
            *self.conversation_history
        ]

        response = self.rag.client.chat.completions.create(
            model=self.rag.llm_model,
            messages=messages
        )

        answer = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": answer})

        self._track_tokens(response.usage)

        return answer, response.usage, self.rag.cumulative_tokens

    def reset(self):
        self.conversation_history = []
        self.rag.cumulative_tokens = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }