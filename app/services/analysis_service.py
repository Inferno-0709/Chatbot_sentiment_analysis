# app/services/analysis_service.py
from typing import Dict, Any, Optional
from transformers import pipeline
from sqlalchemy.orm import Session
from app.models.message_analysis import MessageAnalysis
from app.models.message import Message
import math
import traceback


SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"  



_sentiment_pipe = None
def get_sentiment_pipe():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline("text-classification", model=SENT_MODEL, return_all_scores=True, truncation=True)
    return _sentiment_pipe

def _prob_to_polarity(scores: Dict[str, float]) -> float:
    """
    Map class probability dict (labels -> prob) to continuous polarity [-1 .. 1].
    Expected labels (CardiffNLP): 'negative', 'neutral', 'positive'
    Formula: polarity = p_pos*1 + p_neu*0 + p_neg*(-1)
    """
    p_pos = scores.get("positive", 0.0) or scores.get("POSITIVE", 0.0)
    p_neg = scores.get("negative", 0.0) or scores.get("NEGATIVE", 0.0)
    p_neu = scores.get("neutral", 0.0) or scores.get("NEUTRAL", 0.0)
    s = p_pos + p_neu + p_neg
    if s <= 0:
        return 0.0
    p_pos /= s
    p_neu /= s
    p_neg /= s
    polarity = p_pos * 1.0 + p_neu * 0.0 + p_neg * -1.0
    return float(polarity)


_LABEL_MAP = {
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "POSITIVE": "positive",
}

def _normalize_label(raw_label: str) -> str:
    """
    Convert model-returned label (like 'label_0' or 'LABEL_2' or 'positive') into
    a standardized human-readable label: 'negative'|'neutral'|'positive'.
    Falls back to the raw_label lowercased if unknown.
    """
    if not raw_label:
        return "neutral"
    mapped = _LABEL_MAP.get(raw_label)
    if mapped:
        return mapped
    if raw_label.isdigit():
        idx = int(raw_label)
        if idx == 0:
            return "negative"
        if idx == 1:
            return "neutral"
        if idx == 2:
            return "positive"
    rl = raw_label.lower()
    if "neg" in rl:
        return "negative"
    if "pos" in rl:
        return "positive"
    if "neu" in rl:
        return "neutral"
    return rl  

def analyze_text(text: str) -> Dict[str, Any]:
    """
    Run sentiment model and return structured dict:
      {
         sentiment_label: "positive"/"neutral"/"negative",
         sentiment_score: confidence of top label,
         polarity: continuous score [-1..1],
         raw_scores: raw dict
      }
    This version normalizes model labels like 'LABEL_0' to human names.
    """
    try:
        pipe = get_sentiment_pipe()
        res = pipe(text[:512])[0]  
        scores = {}
        for entry in res:
            raw_label = entry.get("label")
            score = float(entry.get("score", 0.0))
            norm_label = _normalize_label(raw_label)
            scores[norm_label] = scores.get(norm_label, 0.0) + score


        ssum = sum(scores.values()) or 1.0
        for k in list(scores.keys()):
            scores[k] = float(scores[k] / ssum)


        top_label = max(scores.items(), key=lambda x: x[1])[0]
        top_score = scores[top_label]
        polarity = _prob_to_polarity(scores) 
        return {
            "sentiment_label": top_label,
            "sentiment_score": float(top_score),
            "polarity": polarity,
            "raw_scores": scores
        }
    except Exception as e:
        print("Sentiment model error:", e)
        traceback.print_exc()
        return {
            "sentiment_label": "neutral",
            "sentiment_score": 0.5,
            "polarity": 0.0,
            "raw_scores": {}
        }


def analyze_and_store_message(db: Session, message_id: int, text: str):
    """
    Analyze the text and store results into message_analysis table.
    Stores: sentiment_label, sentiment_score (top-class), emotion_label(None), emotion_scores(None),
    and also stores polarity in sentiment_score field (optional). To keep compatibility,
    we fill sentiment_score with top-class confidence, and include polarity in emotion_scores JSON for now.
    """
    data = analyze_text(text)
    analysis = MessageAnalysis(
        message_id=message_id,
        sentiment_label=data["sentiment_label"].upper() if data.get("sentiment_label") else None,
        sentiment_score=float(data.get("sentiment_score", 0.0)),
        emotion_label=None,
        emotion_scores={"polarity": data.get("polarity"), "raw": data.get("raw_scores")}
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis

def compute_user_conversation_sentiment(db: Session, user_id: int):
    """
    Compute conversation-level sentiment for user:
    - Aggregate using mean of per-message polarity (from emotion_scores.polarity)
    - Fall back to mean of sentiment_score if polarity missing.
    Returns float in [-1..1] or None.
    """
    rows = db.query(MessageAnalysis).join(Message, Message.id == MessageAnalysis.message_id).filter(
        Message.user_id == user_id
    ).all()
    if not rows:
        return None
    polarities = []
    for r in rows:
        ev = None
        try:
            ev = r.emotion_scores or {}
            p = ev.get("polarity") if isinstance(ev, dict) else None
            if p is None:
                if r.sentiment_label:
                    lbl = r.sentiment_label.lower()
                    p = 1.0 if "positive" in lbl else (-1.0 if "negative" in lbl else 0.0)
                else:
                    p = 0.0
            polarities.append(float(p))
        except Exception:
            continue
    if not polarities:
        return None
    return float(sum(polarities) / len(polarities))


