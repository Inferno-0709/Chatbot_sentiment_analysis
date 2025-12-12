# app/routers/analytics.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.database import SessionLocal
from app.models.user import User
from app.models.message import Message
from app.models.message_analysis import MessageAnalysis

# Optional: use your llm_service for a nicer textual summary if available
try:
    from app.services.llm_service import generate_reply as llm_generate_reply
    _HAS_LLM = True
except Exception:
    _HAS_LLM = False

router = APIRouter(prefix="/analytics", tags=["Analytics"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- helpers ----------
def _safe_extract_polarity(analysis: Optional[MessageAnalysis]) -> float:
    """
    Extract polarity (float) from a MessageAnalysis row.
    Looks for analysis.emotion_scores['polarity'] (dict) first.
    Falls back to sentiment_label mapping.
    """
    if not analysis:
        return 0.0
    ev = analysis.emotion_scores
    if isinstance(ev, dict):
        p = ev.get("polarity")
        try:
            if p is None:
                raise ValueError("no polarity in dict")
            return float(p)
        except Exception:
            pass
    # fallback to sentiment_label
    lbl = (analysis.sentiment_label or "").lower()
    if "positive" in lbl:
        return 0.8
    if "negative" in lbl:
        return -0.8
    return 0.0

def _moving_average(values: List[float], window: int = 3) -> List[float]:
    if not values:
        return []
    w = max(1, window)
    out = []
    n = len(values)
    for i in range(n):
        start = max(0, i - w + 1)
        window_vals = values[start:i+1]
        out.append(sum(window_vals) / len(window_vals))
    return out

def _linear_regression_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs)/n
    mean_y = sum(ys)/n
    num = sum((xs[i]-mean_x)*(ys[i]-mean_y) for i in range(n))
    den = sum((xs[i]-mean_x)**2 for i in range(n))
    if den == 0:
        return None
    return num/den

def _label_trend(slope: Optional[float], delta: float, thresholds: Dict[str,float] = None) -> str:
    if thresholds is None:
        thresholds = {"slope_small": 0.01, "delta_big": 0.25}
    if slope is None:
        return "stable"
    if slope > thresholds["slope_small"] or delta > thresholds["delta_big"]:
        return "increasing"
    if slope < -thresholds["slope_small"] or delta < -thresholds["delta_big"]:
        return "decreasing"
    return "stable"

def _polarity_to_word(score: Optional[float]) -> str:
    if score is None:
        return "Unknown"
    try:
        s = float(score)
    except Exception:
        return "Unknown"
    if s <= -0.6:
        return "Strongly Negative"
    elif s <= -0.2:
        return "Negative"
    elif s < 0.2:
        return "Neutral"
    elif s < 0.6:
        return "Positive"
    else:
        return "Strongly Positive"

# ---------- endpoints ----------
@router.get("/user/{user_id}/sentiment")
def conversation_sentiment(user_id: int, db: Session = Depends(get_db)):
    """
    Returns aggregated conversation-level sentiment for the user.
    Output JSON:
    {
      "user_id": int,
      "conversation_sentiment": float or null,   # aggregated polarity [-1..1]
      "label": "Neutral"/"Positive"/...,
      "count": number_of_analyzed_messages
    }
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="user not found")

    # fetch analyses joined with messages to ensure correct user
    rows = db.query(MessageAnalysis).join(Message, Message.id == MessageAnalysis.message_id).filter(
        Message.user_id == user_id
    ).all()

    if not rows:
        return {"user_id": user_id, "conversation_sentiment": None, "label": "Unknown", "count": 0}

    polarities = []
    for r in rows:
        try:
            p = _safe_extract_polarity(r)
            polarities.append(float(p))
        except Exception:
            continue

    if not polarities:
        return {"user_id": user_id, "conversation_sentiment": None, "label": "Unknown", "count": 0}

    agg = float(sum(polarities) / len(polarities))
    label = _polarity_to_word(agg)
    return {"user_id": user_id, "conversation_sentiment": agg, "label": label, "count": len(polarities)}

@router.get("/user/{user_id}/mood_trend")
def user_mood_trend(user_id: int, db: Session = Depends(get_db), window: int = 3, last_n: int = 200):
    """
    Returns a richer mood-trend analysis for the user's recent conversation.
    Response JSON includes numeric arrays and a short summary string.

    Query params:
      - window: smoothing window for moving average (default 3)
      - last_n: number of most recent messages to consider (default 200)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="user not found")

    # fetch last messages (newest->oldest)
    msgs = db.query(Message).filter(Message.user_id == user_id).order_by(Message.created_at.desc()).limit(last_n).all()
    if not msgs:
        return {
            "user_id": user_id,
            "count": 0,
            "polarities": [],
            "smoothed": [],
            "slope": None,
            "start_mean": None,
            "end_mean": None,
            "delta": None,
            "trend": "unknown",
            "shift_points": [],
            "summary": ""
        }

    msgs = list(reversed(msgs))  # now oldest->newest

    polarities: List[float] = []
    msg_meta: List[Dict[str, Any]] = []

    for m in msgs:
        analysis = db.query(MessageAnalysis).filter(MessageAnalysis.message_id == m.id).first()
        p = None
        if analysis and analysis.emotion_scores:
            ev = analysis.emotion_scores
            if isinstance(ev, dict):
                p = ev.get("polarity")
        if p is None:
            # fallback mapping
            lbl = (analysis.sentiment_label or "").lower() if analysis else ""
            if "positive" in lbl:
                p = 0.8
            elif "negative" in lbl:
                p = -0.8
            else:
                p = 0.0
        try:
            p = float(p)
        except Exception:
            p = 0.0
        polarities.append(p)
        msg_meta.append({"message_id": m.id, "created_at": m.created_at.isoformat() if m.created_at else None})

    smoothed = _moving_average(polarities, window=window)
    xs = list(range(len(smoothed)))
    slope = _linear_regression_slope(xs, smoothed)

    w = max(1, window)
    start_slice = smoothed[:w] if len(smoothed) >= w else smoothed[: max(1, len(smoothed))]
    end_slice = smoothed[-w:] if len(smoothed) >= w else smoothed[- max(1, len(smoothed))]
    start_mean = float(sum(start_slice) / len(start_slice)) if start_slice else 0.0
    end_mean = float(sum(end_slice) / len(end_slice)) if end_slice else 0.0
    delta = end_mean - start_mean

    trend_label = _label_trend(slope, delta)

    # detect shift points
    shift_points: List[Dict[str, Any]] = []
    for i in range(1, len(smoothed)):
        prev = smoothed[i-1]
        cur = smoothed[i]
        # crossing zero or large jump
        if (prev <= 0 and cur > 0) or (prev >= 0 and cur < 0):
            shift_points.append({
                "index": i,
                "message_id": msg_meta[i]["message_id"],
                "timestamp": msg_meta[i]["created_at"],
                "polarity": cur,
                "reason": "crossed_zero"
            })
        if abs(cur - prev) >= 0.5:
            shift_points.append({
                "index": i,
                "message_id": msg_meta[i]["message_id"],
                "timestamp": msg_meta[i]["created_at"],
                "polarity": cur,
                "reason": "large_jump"
            })

    result = {
        "user_id": user_id,
        "count": len(polarities),
        "polarities": polarities,
        "smoothed": smoothed,
        "slope": slope,
        "start_mean": start_mean,
        "end_mean": end_mean,
        "delta": delta,
        "trend": trend_label,
        "shift_points": shift_points,
    }

    # textual summary: simple template by default
    simple_summary = f"Conversation mood is {trend_label}. Start mean={start_mean:+.2f}, end mean={end_mean:+.2f}, delta={delta:+.2f}."
    if shift_points:
        simple_summary += f" Detected {len(shift_points)} notable shift(s) (examples: {', '.join(sp['reason'] for sp in shift_points[:3])})."

    summary_text = simple_summary

    # If LLM available, ask for a concise human-friendly summary
    if _HAS_LLM:
        try:
            prompt = (
                "You are a helpful assistant that summarizes mood trends.\n\n"
                f"Inputs:\n- trend: {trend_label}\n- start_mean: {start_mean:.2f}\n- end_mean: {end_mean:.2f}\n- delta: {delta:.2f}\n"
                f"- slope: {slope if slope is not None else 'N/A'}\n- detected_shifts: {len(shift_points)} (reasons: {[sp['reason'] for sp in shift_points[:5]]})\n\n"
                "Write a concise human-friendly summary (2-3 sentences) describing how the user's mood changed across the conversation and suggested next steps for the assistant if any."
            )
            # use llm_generate_reply with empty history + prompt as user_message
            llm_resp = llm_generate_reply(history="", user_message=prompt)
            if isinstance(llm_resp, str) and llm_resp.strip():
                summary_text = llm_resp.strip()
        except Exception:
            # ignore LLM errors and keep simple summary
            pass

    result["summary"] = summary_text
    result["summary_label"] = _polarity_to_word(end_mean)

    return result
