from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from api.models import SettingsPayload
from api.utils import save_settings, get_settings, build_chatbot

router = APIRouter()

@router.get("/settings")
def read_settings():
    return get_settings()

@router.post("/settings")
def write_settings(payload: SettingsPayload, request: Request):
    save_settings(payload.vault_path, payload.exclude_folders, payload.knowledge_base)

    try:
        embed_model = request.app.state.embed_model
        redis_client = request.app.state.redis_client
        request.app.state.chatbot = build_chatbot(embed_model, redis_client)
    except Exception as e:
        raise HTTPException(500, f"Failed to rebuild knowledge base: {e}")

    return {"status": "ok"}

@router.post("/settings/rebuild")
def rebuild(request: Request):
    s = get_settings()
    kb_path = Path(s["knowledge_base"])

    if kb_path.exists():
        kb_path.unlink()

    try:
        embed_model = request.app.state.embed_model
        redis_client = request.app.state.redis_client
        request.app.state.chatbot = build_chatbot(embed_model, redis_client)
    except Exception as e:
        raise HTTPException(500, f"Rebuild failed: {e}")

    return {"status": "ok"}