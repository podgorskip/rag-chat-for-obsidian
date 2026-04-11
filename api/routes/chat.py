from fastapi import APIRouter, HTTPException, Depends
from api.dependencies import get_chatbot
from api.models import MessageResponse, MessageRequest
from chatbot.chatbot import Chatbot

router = APIRouter()

@router.post("/chat", response_model=MessageResponse)
def chat(
        req: MessageRequest,
        chatbot: Chatbot = Depends(get_chatbot)
):
    if chatbot is None:
        raise HTTPException(503, "Chatbot not initialized.")

    confirm_external = (lambda: False) if req.allow_external is None else (lambda: req.allow_external)
    result = chatbot.chat(req.message, session_id=req.session_id, confirm_external=confirm_external)

    if result is None:
        return MessageResponse(needs_confirmation=True)

    return MessageResponse(
        answer=result["answer"],
        thinking=result["thinking"],
        sources=result["sources"],
        total_tokens=result["tokens"]["total_tokens"]
    )