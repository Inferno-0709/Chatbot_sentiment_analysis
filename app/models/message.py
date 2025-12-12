# app/models/message.py
import datetime
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, ForeignKey
from app.database import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # <-- changed
    sender = Column(String, nullable=False)  # 'user' or 'bot'
    text = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)
