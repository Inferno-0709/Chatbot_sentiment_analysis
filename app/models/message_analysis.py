# app/models/message_analysis.py
import datetime
from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, Float
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from app.database import Base

class MessageAnalysis(Base):
    __tablename__ = "message_analysis"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    sentiment_label = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    emotion_label = Column(String, nullable=True)
    emotion_scores = Column(SQLITE_JSON, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)
