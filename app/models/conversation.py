# app/models/conversation.py
import datetime
from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey
from app.database import Base

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)
