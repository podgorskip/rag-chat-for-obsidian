from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from sentence_transformers import SentenceTransformer
from api.utils import build_chatbot, get_redis_client
from chatbot.chatbot import Chatbot
from api.routes import chat, reset, settings, sessions
import redis as redis_lib

chatbot: Chatbot | None = None
embed_model: SentenceTransformer | None = None
redis_client: redis_lib.Redis | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot, embed_model, redis_client
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    redis_client = get_redis_client()
    app.state.redis_client = redis_client
    app.state.embed_model = embed_model
    app.state.chatbot = build_chatbot(embed_model, redis_client)

    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(chat.router)
app.include_router(reset.router)
app.include_router(settings.router)
app.include_router(sessions.router)

@app.get("/")
def ui():
    return FileResponse("templates/index.html")