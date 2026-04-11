import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from redis import Redis
import redis as redis_lib
from chatbot.chatbot import Chatbot
from connectors.vault_connector import build_knowledge_base
from embedders.embedder import Embedder
from embedders.embedder_config import EmbedderConfig
from rags.llm_client import LLMClient
from rags.rag import RAG

load_dotenv()

def get_settings() -> dict:
    load_dotenv(override=True)
    return {
        "vault_path":       os.getenv("VAULT_PATH", ""),
        "exclude_folders":  os.getenv("EXCLUDE_FOLDERS", ""),
        "knowledge_base":   os.getenv("KNOWLEDGE_BASE", "generated_sources/knowledge_base.pkl"),
    }

def save_settings(vault_path: str, exclude_folders: str, knowledge_base: str):
    env_path = Path(".env")
    lines = env_path.read_text().splitlines() if env_path.exists() else []

    updates = {
        "VAULT_PATH":       vault_path,
        "EXCLUDE_FOLDERS":  exclude_folders,
        "KNOWLEDGE_BASE":   knowledge_base,
    }

    existing_keys = set()
    new_lines = []
    for line in lines:
        key = line.split("=")[0].strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            existing_keys.add(key)
        else:
            new_lines.append(line)

    for key, val in updates.items():
        if key not in existing_keys:
            new_lines.append(f"{key}={val}")

    env_path.write_text("\n".join(new_lines) + "\n")
    load_dotenv(override=True)


def build_chatbot(redis_client) -> Chatbot:
    s = get_settings()
    kb_path = s["knowledge_base"]
    exclude = [f.strip() for f in s["exclude_folders"].split(",") if f.strip()]
    cfg = EmbedderConfig(provider="bge", model_name="BAAI/bge-m3")
    embedder = Embedder(cfg)

    if Path(kb_path).exists():
        df = pd.read_pickle(kb_path)
    else:
        Path(kb_path).parent.mkdir(parents=True, exist_ok=True)
        df = build_knowledge_base(
            vault_path=s["vault_path"],
            exclude_folders=exclude,
            output_path=kb_path,
            embedder=embedder
        )

    client = LLMClient(provider="ollama", model="llama3.2")
    rag    = RAG(client=client, embedder=embedder, df=df, llm_model="llama3.2")
    return Chatbot(rag, redis_client)

def get_redis_client() -> Redis:
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis_lib.from_url(REDIS_URL, decode_responses=True)