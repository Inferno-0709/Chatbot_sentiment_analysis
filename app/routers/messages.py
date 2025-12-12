# app/routers/messages.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import SessionLocal
from app.models.message import Message
from app.models.message_analysis import MessageAnalysis
from app.models.user import User
from app.schemas.message import MessageOut, MessageAnalysisOut, MessageWithAnalysis

router = APIRouter(prefix="/messages", tags=["Messages"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/{message_id}/analysis", response_model=MessageAnalysisOut)
def get_message_analysis(message_id: int, db: Session = Depends(get_db)):
    """
    Return the analysis row for a given message_id.
    404 if message or analysis not found.
    """
    msg = db.query(Message).filter(Message.id == message_id).first()
    if not msg:
        raise HTTPException(status_code=404, detail="message not found")

    analysis = db.query(MessageAnalysis).filter(MessageAnalysis.message_id == message_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="analysis not found for this message")
    return analysis

@router.get("/user/{user_id}", response_model=List[MessageWithAnalysis])
def get_user_messages_with_analysis(user_id: int, limit: Optional[int] = 100, db: Session = Depends(get_db)):
    """
    Return the last `limit` messages for a user together with their analysis (if available),
    ordered newest->oldest.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="user not found")

    messages = db.query(Message).filter(Message.user_id == user_id).order_by(Message.created_at.desc()).limit(limit).all()

    out = []
    for m in messages:
        analysis = db.query(MessageAnalysis).filter(MessageAnalysis.message_id == m.id).first()
        out.append({"message": m, "analysis": analysis})
    return out
