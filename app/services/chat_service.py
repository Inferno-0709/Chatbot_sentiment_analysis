# app/services/chat_service.py
import traceback
from typing import Tuple

from sqlalchemy.orm import Session

from app.models.message import Message
from app.models.message_analysis import MessageAnalysis
from app.services.analysis_service import analyze_and_store_message
from app.services.llm_service import generate_reply


def _build_history_text(db: Session, user_id: int, max_messages: int = 12) -> str:
    """
    Fetch last max_messages (newest->oldest) for this user and return as
    a plain-text transcript oldest->newest with role labels.
    """
    msgs = (
        db.query(Message)
        .filter(Message.user_id == user_id)
        .order_by(Message.created_at.desc())
        .limit(max_messages)
        .all()
    )
    msgs = list(reversed(msgs))
    lines = []
    for m in msgs:
        role = "User" if m.sender == "user" else "Assistant"
        text = (m.text or "").strip()
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def process_chat(db: Session, user_id: int, text: str) -> Tuple[int, int, str]:
    """
    Save user message, analyze it, call the LLM with recent context, save bot reply.
    Returns: (user_message_id, bot_message_id, bot_reply)
    """
    user_msg = Message(user_id=user_id, sender="user", text=text)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)


    try:
        analyze_and_store_message(db=db, message_id=user_msg.id, text=text)
    except Exception as e:
        print("Error during analyze_and_store_message:", e)
        traceback.print_exc()
        try:
            fallback = MessageAnalysis(
                message_id=user_msg.id,
                sentiment_label="NEUTRAL",
                sentiment_score=0.5
            )
            db.add(fallback)
            db.commit()
            db.refresh(fallback)
            print("Stored fallback analysis for message", user_msg.id)
        except Exception as ex:
            print("Failed to store fallback analysis:", ex)
            traceback.print_exc()

    try:
        history_text = _build_history_text(db=db, user_id=user_id, max_messages=12)
        bot_reply = generate_reply(history=history_text, user_message=text)
        if bot_reply is None:
            bot_reply = f"(llm-empty) You said: {text}"
    except Exception as e:
        print("LLM generation failed:", e)
        traceback.print_exc()
        bot_reply = f"(llm-error) You said: {text}"

    bot_msg = Message(user_id=user_id, sender="bot", text=bot_reply)
    db.add(bot_msg)
    db.commit()
    db.refresh(bot_msg)

    return user_msg.id, bot_msg.id, bot_reply
