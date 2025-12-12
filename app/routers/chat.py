# app/routers/chat.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import process_chat
from app.models.message_analysis import MessageAnalysis

router = APIRouter(prefix="/chat", tags=["Chat"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    user_msg_id, bot_msg_id, reply = process_chat(db=db, user_id=payload.user_id, text=payload.text)

    
    analysis = db.query(MessageAnalysis).filter(MessageAnalysis.message_id == user_msg_id).first()
    return ChatResponse(
        user_message_id=user_msg_id,
        bot_message_id=bot_msg_id,
        bot_reply=reply,
        analysis=analysis
    )
