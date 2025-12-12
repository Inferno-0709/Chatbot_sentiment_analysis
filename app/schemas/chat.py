# app/schemas/chat.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_id: int
    text: str

class ChatResponse(BaseModel):
    user_message_id: int
    bot_message_id: int
    bot_reply: str
