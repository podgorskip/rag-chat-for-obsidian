import json
import logging
import re
import textwrap
import time
import uuid
from typing import Callable
import redis
from rags.rag import RAG
from rags.utils import format_history

class Chatbot:
    SESSIONS_ZSET = "chat:sessions"

    def __init__(self, rag: RAG, redis_client: redis.Redis):
        self.rag = rag
        self.redis = redis_client
        self.rag.cumulative_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _hkey(self, sid: str) -> str: return f"chat:history:{sid}"
    def _mkey(self, sid: str) -> str: return f"chat:meta:{sid}"

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        now = time.time()
        meta = {"session_id": sid, "title": "New Chat", "created_at": now, "updated_at": now}
        self.redis.set(self._mkey(sid), json.dumps(meta))
        self.redis.zadd(self.SESSIONS_ZSET, {sid: now})
        return sid

    def list_sessions(self) -> list[dict]:
        sids = self.redis.zrevrange(self.SESSIONS_ZSET, 0, -1)
        result = []
        for sid in sids:
            raw = self.redis.get(self._mkey(sid))
            if raw:
                result.append(json.loads(raw))
        return result

    def delete_session(self, session_id: str) -> None:
        self.redis.delete(self._hkey(session_id))
        self.redis.delete(self._mkey(session_id))
        self.redis.zrem(self.SESSIONS_ZSET, session_id)

    def get_display_history(self, session_id: str) -> list[dict]:
        history = self._load_history(session_id)
        visible = []
        for turn in history:
            if turn["role"] == "user":
                visible.append({"role": "user", "content": turn.get("display", turn["content"])})
            else:
                visible.append({"role": "assistant", "content": turn["content"]})
        return visible

    def _load_history(self, session_id: str) -> list[dict]:
        raw = self.redis.get(self._hkey(session_id))
        return json.loads(raw) if raw else []

    def _save_history(self, session_id: str, history: list[dict]) -> None:
        self.redis.set(self._hkey(session_id), json.dumps(history))

    def _touch_meta(self, session_id: str, title: str | None = None) -> None:
        now = time.time()
        raw = self.redis.get(self._mkey(session_id))
        meta = json.loads(raw) if raw else {
            "session_id": session_id, "title": "New Chat", "created_at": now
        }
        meta["updated_at"] = now
        if title and meta["title"] == "New Chat":
            meta["title"] = title
        self.redis.set(self._mkey(session_id), json.dumps(meta))
        self.redis.zadd(self.SESSIONS_ZSET, {session_id: now})

    def chat(
            self,
            query: str,
            session_id: str,
            confirm_external: Callable[[], bool] | None = None,
    ):
        history = self._load_history(session_id)
        history_str = format_history(history)
        title = textwrap.shorten(query, width=60, placeholder="…")
        self._touch_meta(session_id, title)

        standalone_query = self.rag.process_query(query, history_str, self.rag.config.contextualize_prompt)
        query_embedding = self.rag.embed_query(standalone_query)
        chunks = self.rag.retrieve(query, query_embedding)

        if not chunks:
            logging.debug("Fallback retrieval returned no chunks for session=%s", session_id)

        context = self.rag.build_context(chunks) if chunks else ""

        if not context:
            if confirm_external is None or not confirm_external():
                return None
            system_prompt = self.rag.config.external_prompt
            logging.info("Using external prompt for session=%s, prompt")
        else:
            system_prompt = self.rag.config.answer_prompt.format(
                context=context or "No context available."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            *[{"role": t["role"], "content": t["content"]} for t in history],
            {"role": "user", "content": standalone_query},
        ]

        try:
            response = self.rag.client.chat.completions.create(
                model=self.rag.llm_model,
                messages=messages,
            )
        except Exception as e:
            raise RuntimeError(f"LLM call failed for session={session_id!r}: {e}") from e

        raw_response = response.choices[0].message.content
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_response, re.DOTALL)
        answer = answer_match.group(1) if answer_match else None
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw_response, re.DOTALL)
        thinking = thinking_match.group(1) if thinking_match else None

        history.append({"role": "user", "content": standalone_query})
        history.append({"role": "assistant", "content": answer})
        self._save_history(session_id, history)

        self._touch_meta(session_id)
        self._track_tokens(response.usage)
        self._log(standalone_query, context, answer, thinking)

        return {
            "answer": answer,
            "tokens": self.rag.cumulative_tokens,
            "thinking": thinking,
            "sources": [
                {"title": chunk["title"], "source": chunk["content"]}
                for chunk in chunks
            ]
        }

    def _track_tokens(self, usage) -> None:
        if self.rag.client.provider != "openai" or not usage:
            return
        self.rag.cumulative_tokens["prompt_tokens"]     += usage.prompt_tokens
        self.rag.cumulative_tokens["completion_tokens"] += usage.completion_tokens
        self.rag.cumulative_tokens["total_tokens"]      += usage.total_tokens

    def _log(self, query: str, context: str, answer: str, thinking: str) -> None:
        logging.info("[QUESTION]: %s", query)
        logging.info("[CONTEXT]: %s", context)
        logging.info("[ANSWER]: %s", answer)
        logging.info("[THINKING]: %s", thinking)