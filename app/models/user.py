# app/models/user.py
import datetime
from sqlalchemy import Column, Integer, String, TIMESTAMP
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)
