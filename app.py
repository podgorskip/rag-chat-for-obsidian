from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import logging
import os
from sentence_transformers import SentenceTransformer
from rags.rag import RAG
from chatbot.chatbot import Chatbot
from rags.llm_client import LLMClient
from connectors.obsidian_connector import build_knowledge_base

load_dotenv()

chatbot: Chatbot | None = None
embed_model: SentenceTransformer | None = None

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


def build_chatbot() -> Chatbot:
    s = get_settings()
    kb_path = s["knowledge_base"]
    exclude = [f.strip() for f in s["exclude_folders"].split(",") if f.strip()]

    if Path(kb_path).exists():
        df = pd.read_pickle(kb_path)
    else:
        Path(kb_path).parent.mkdir(parents=True, exist_ok=True)
        df = build_knowledge_base(
            vault_path=s["vault_path"],
            exclude_folders=exclude,
            output_path=kb_path,
        )

    client = LLMClient(provider="ollama", model="llama3.2")
    rag    = RAG(client=client, embedding_model=embed_model, df=df, llm_model="llama3.2")
    return Chatbot(rag)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot, embed_model
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chatbot = build_chatbot()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    answer: str
    total_tokens: int

class SettingsPayload(BaseModel):
    vault_path:      str
    exclude_folders: str
    knowledge_base:  str

@app.get("/")
def ui():
    return FileResponse("templates/index.html")

@app.post("/chat", response_model=MessageResponse)
def chat(req: MessageRequest):
    if chatbot is None:
        raise HTTPException(503, "Chatbot not initialized.")
    result = chatbot.chat(req.message)
    if result is None:
        raise HTTPException(404, "No relevant context found.")
    answer, _, tokens = result
    return MessageResponse(answer=answer, total_tokens=tokens["total_tokens"])

@app.post("/reset")
def reset():
    if chatbot is None:
        raise HTTPException(503, "Chatbot not initialized.")
    chatbot.reset()
    return {"status": "ok"}

@app.get("/settings")
def read_settings():
    return get_settings()

@app.post("/settings")
def write_settings(payload: SettingsPayload):
    global chatbot
    save_settings(payload.vault_path, payload.exclude_folders, payload.knowledge_base)
    try:
        chatbot = build_chatbot()
    except Exception as e:
        raise HTTPException(500, f"Failed to rebuild knowledge base: {e}")
    return {"status": "ok"}

@app.post("/settings/rebuild")
def rebuild():
    global chatbot
    s = get_settings()
    kb_path = Path(s["knowledge_base"])
    if kb_path.exists():
        kb_path.unlink()
    try:
        chatbot = build_chatbot()
    except Exception as e:
        raise HTTPException(500, f"Rebuild failed: {e}")
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok", "ready": chatbot is not None}