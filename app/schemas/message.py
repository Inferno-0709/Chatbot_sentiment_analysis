# app/schemas/message.py
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class MessageOut(BaseModel):
    id: int
    user_id: int
    sender: str
    text: str
    created_at: Optional[datetime]

    class Config:
        orm_mode = True

class MessageAnalysisOut(BaseModel):
    id: int
    message_id: int
    sentiment_label: Optional[str]
    sentiment_score: Optional[float]
    emotion_label: Optional[str] = None
    emotion_scores: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime]

    class Config:
        orm_mode = True

class MessageWithAnalysis(BaseModel):
    message: MessageOut
    analysis: Optional[MessageAnalysisOut]
