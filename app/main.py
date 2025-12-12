# app/main.py
from fastapi import FastAPI
from app.database import Base, engine

# import models so SQLAlchemy registers them
import app.models.user
import app.models.message
import app.models.message_analysis

from app.routers import users, chat, analytics, messages

app = FastAPI(title="Simplified Chatbot - user_id only")

# create all tables
Base.metadata.create_all(bind=engine)

# include routers
app.include_router(users.router)
app.include_router(chat.router)
app.include_router(analytics.router)
app.include_router(messages.router)

@app.get("/")
def root():
    return {"status": "ok", "message": "Simplified chat backend running (user_id only)."}
