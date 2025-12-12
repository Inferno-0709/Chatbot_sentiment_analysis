# gradio_app.py
"""
Full Gradio UI for Chatbot with:
 - login
 - chat with per-message sentiment inline
 - conversation sentiment label (word + score)
 - mood trend chart (raw + smoothed) and human-friendly summary
 - auto-clear + autofocus message box after send
 - dynamic/responsive right-panel (CSS + JS)
 - safe refresh + Interval auto-update (if Gradio supports it)

Run:
 python gradio_app.py
"""

import io
import requests
import gradio as gr
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Any
from PIL import Image

# ----------------- Config / Endpoints -----------------
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
USERS_ENDPOINT = "/users/"
CHAT_ENDPOINT = "/chat/"
MESSAGES_ENDPOINT = "/messages/user/{user_id}?limit={limit}"
SENTIMENT_ENDPOINT = "/analytics/user/{user_id}/sentiment"
MOOD_TREND_ENDPOINT = "/analytics/user/{user_id}/mood_trend"

# ----------------- Backend helpers -----------------
def create_or_get_user(backend_url: str, username: str) -> Dict:
    url = backend_url.rstrip("/") + USERS_ENDPOINT
    r = requests.post(url, json={"username": username}, timeout=10)
    r.raise_for_status()
    return r.json()

def post_chat(backend_url: str, user_id: int, text: str) -> Dict:
    url = backend_url.rstrip("/") + CHAT_ENDPOINT
    r = requests.post(url, json={"user_id": user_id, "text": text}, timeout=25)
    r.raise_for_status()
    return r.json()

def fetch_user_messages(backend_url: str, user_id: int, limit: int = 200) -> List[Dict]:
    url = backend_url.rstrip("/") + MESSAGES_ENDPOINT.format(user_id=user_id, limit=limit)
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    return r.json()

def get_user_sentiment(backend_url: str, user_id: int) -> Optional[float]:
    url = backend_url.rstrip("/") + SENTIMENT_ENDPOINT.format(user_id=user_id)
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("conversation_sentiment", None)

def fetch_mood_trend(backend_url: str, user_id: int, window: int = 3, last_n: int = 200) -> Dict[str, Any]:
    url = backend_url.rstrip("/") + MOOD_TREND_ENDPOINT.format(user_id=user_id)
    params = {"window": window, "last_n": last_n}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# ----------------- UI helpers -----------------
def build_history_from_messages(messages_with_analysis: List[Dict]) -> List[Dict]:
    """
    Convert server /messages/user/{id} response to Gradio chat format.
    Backend message objects expected: {"message": {...}, "analysis": {...}}
    We show per-message sentiment inline for user messages.
    """
    out = []
    for item in reversed(messages_with_analysis):
        m = item.get("message", {})
        a = item.get("analysis")
        role = "user" if (m.get("sender") == "user" or m.get("sender") == "User") else "assistant"
        text = (m.get("text") or "").strip()
        if role == "user":
            if a:
                label = a.get("sentiment_label", "NEUTRAL")
                score = a.get("sentiment_score")
                if score is not None:
                    text = f"{text}   —   [{label.upper()} {float(score):.2f}]"
                else:
                    text = f"{text}   —   [{label.upper()}]"
            else:
                text = f"{text}   —   [PENDING]"
        out.append({"role": role, "content": text})
    return out

def polarity_to_word(score: Optional[float]) -> str:
    """Map numeric polarity [-1..1] to human-friendly label."""
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

def plot_mood_trend_image(polarities: List[float], smoothed: List[float], shift_points: List[Dict[str,Any]]):
    """Produce PNG bytes of the mood trend plot (raw + smoothed + shift markers)."""
    fig, ax = plt.subplots(figsize=(6,3.5))
    if polarities:
        ax.plot(range(len(polarities)), polarities, label="raw", linewidth=1, alpha=0.7)
    if smoothed:
        ax.plot(range(len(smoothed)), smoothed, label="smoothed", linewidth=2)
    if shift_points:
        xs = [sp["index"] for sp in shift_points if sp.get("index") is not None]
        ys = [sp["polarity"] for sp in shift_points if sp.get("polarity") is not None]
        if xs and ys:
            ax.scatter(xs, ys, marker='o', zorder=5)
    ax.axhline(0.0, linestyle='--', linewidth=0.8, color="gray")
    ax.set_xlabel("Message # (oldest → newest)")
    ax.set_ylabel("Polarity [-1..1]")
    ax.set_title("Mood Trend")
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ----------------- Interaction helpers -----------------
def refresh_sentiment_button(backend_url: Optional[str] = None, logged_in_state: Optional[dict] = None):
    """
    Safe refresh for the conversation sentiment label.

    Accepts optional args so it won't crash if it's called without inputs.
    Returns a friendly string to display in the sentiment textbox.
    """
    if not backend_url or not logged_in_state or not isinstance(logged_in_state, dict):
        return "Not logged in"

    user_id = logged_in_state.get("user_id")
    if user_id is None:
        return "Not logged in"

    try:
        conv = get_user_sentiment(backend_url, user_id)
    except Exception as e:
        return f"Refresh error: {e}"

    if conv is None:
        return "Unknown"
    return f"{polarity_to_word(conv)} ({conv:+.2f})"


# ----------------- App functions -----------------
def login(backend_url: str, username: str):
    """Create or fetch user from backend."""
    if not username or username.strip() == "":
        return "Enter a username", None
    try:
        user = create_or_get_user(backend_url, username.strip())
        return f"Logged in as {user.get('username')} (id: {user.get('id')})", {"user_id": user.get("id"), "username": user.get("username")}
    except Exception as e:
        return f"Login failed: {e}", None

def load_history(backend_url: str, logged_in_state, limit: int = 200):
    """Load message history and conversation sentiment."""
    if not logged_in_state or not isinstance(logged_in_state, dict):
        return [], "Not logged in"
    user_id = logged_in_state.get("user_id")
    if user_id is None:
        return [], "Not logged in"
    try:
        messages = fetch_user_messages(backend_url, user_id, limit=limit)
        history = build_history_from_messages(messages)
    except Exception:
        history = []
    try:
        conv_sent = get_user_sentiment(backend_url, user_id)
    except Exception:
        conv_sent = None
    sentiment_text = f"{polarity_to_word(conv_sent)} ({conv_sent:+.2f})" if conv_sent is not None else "Unknown"
    return history, sentiment_text

def send_message(backend_url: str, logged_in_state, message: str, limit: int = 200):
    """
    Send message -> rebuild history -> refresh sentiment.
    Returns (chat_history, sentiment_text, cleared_input)
    """
    if not logged_in_state or not isinstance(logged_in_state, dict):
        return [], "Not logged in", ""

    user_id = logged_in_state.get("user_id")
    if not user_id:
        return [], "Not logged in", ""

    
    if not message or message.strip() == "":
        try:
            messages = fetch_user_messages(backend_url, user_id, limit=limit)
            history = build_history_from_messages(messages)
            conv_sent = get_user_sentiment(backend_url, user_id)
            sentiment_text = f"{polarity_to_word(conv_sent)} ({conv_sent:+.2f})" if conv_sent is not None else "Unknown"
            return history, sentiment_text, ""
        except Exception as e:
            return [], f"Refresh failed: {e}", ""

    
    try:
        post_chat(backend_url, user_id, message)
    except Exception as e:
        
        try:
            messages = fetch_user_messages(backend_url, user_id, limit=limit)
            history = build_history_from_messages(messages)
        except Exception:
            history = []
        return history, f"Send failed: {e}", ""

    
    try:
        messages = fetch_user_messages(backend_url, user_id, limit=limit)
        history = build_history_from_messages(messages)
    except Exception as e:
        return [], f"Refresh after send failed: {e}", ""

    try:
        conv_sent = get_user_sentiment(backend_url, user_id)
    except Exception:
        conv_sent = None
    sentiment_text = f"{polarity_to_word(conv_sent)} ({conv_sent:+.2f})" if conv_sent is not None else "Unknown"
    return history, sentiment_text, ""

def clear_chat():
    """Clear chat UI only."""
    return [], "Not logged in"

def get_and_plot_mood(backend_url: str, logged_in_state, window: int = 3, last_n: int = 200):
    """
    Fetch mood_trend endpoint and return (PIL.Image, summary_text, sentiment_label_text).
    """
    if not logged_in_state or not isinstance(logged_in_state, dict):
        return None, "Not logged in", "Unknown"
    user_id = logged_in_state.get("user_id")
    if user_id is None:
        return None, "Not logged in", "Unknown"

    try:
        payload = fetch_mood_trend(backend_url, user_id, window=window, last_n=last_n)
    except Exception as e:
        return None, f"Failed to fetch mood trend: {e}", "Unknown"

    polarities = payload.get("polarities", []) or []
    smoothed = payload.get("smoothed", []) or []
    shift_points = payload.get("shift_points", []) or []
    summary = payload.get("summary", "") or ""
    end_mean = payload.get("end_mean")
    label = polarity_to_word(end_mean)

    try:
        img_bytes = plot_mood_trend_image(polarities, smoothed, shift_points)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return None, f"Plotting failed: {e}", f"{label} ({end_mean:+.2f})" if end_mean is not None else label

    label_text = f"{label} ({end_mean:+.2f})" if end_mean is not None else label
    return img, summary, label_text

# ----------------- CSS + JS injection for dynamic right panel -----------------
demo_css_js = gr.HTML(
    """
    <style>
    /* right panel flex layout */
    .right-panel { display: flex; flex-direction: column; gap: 10px; align-items: stretch; }

    /* allow components to shrink/grow */
    .right-panel .gr-box { flex: 0 0 auto; }
    .right-panel .gr-textbox, .right-panel .gr-text-area { flex: 0 0 auto; }

    /* summary box takes remaining vertical space and can grow */
    .right-panel .summary-box { flex: 1 1 auto; min-height: 80px; max-height: 80vh; overflow: auto; }
    .right-panel .summary-box textarea, .right-panel .summary-box .output_text { 
      min-height: 80px; max-height: 80vh; overflow: auto; white-space: pre-wrap;
    }

    /* sentiment box small but flexible */
    .right-panel .sentiment-box { flex: 0 1 auto; min-height: 36px; max-height: 200px; overflow: auto; }
    .right-panel .sentiment-box textarea, .right-panel .sentiment-box .output_text {
      min-height: 36px; max-height: 200px; overflow: auto; white-space: pre-wrap;
    }

    /* responsive image */
    .right-panel img { max-width: 100%; height: auto; display: block; }
    .right-panel .gr-image { flex: 0 0 auto; }

    /* ensure wrapping and prevent clipping */
    .right-panel .gr-box .output_text, .right-panel .gr-box textarea {
        white-space: pre-wrap; word-wrap: break-word;
    }

    @media (max-width: 900px) {
      .right-panel { gap: 8px; }
    }
    </style>

    <script>
    // Auto-resize function for textareas and Gradio output divs inside right-panel
    function autosizeRightPanelTextareasAndDivs() {
      const container = document.querySelector('.right-panel');
      if (!container) return;

      // Resize textareas
      const textareas = container.querySelectorAll('textarea');
      textareas.forEach(ta => {
        ta.style.height = 'auto';
        ta.style.height = (ta.scrollHeight) + 'px';
        if (!ta._autosize_attached) {
          ta.addEventListener('input', () => {
            ta.style.height = 'auto';
            ta.style.height = (ta.scrollHeight) + 'px';
          });
          ta._autosize_attached = true;
        }
      });

      // Resize Gradio output divs (class names vary)
      const outputDivs = container.querySelectorAll('.output_text, .gr-output, .gradio-output');
      outputDivs.forEach(d => {
        d.style.whiteSpace = 'pre-wrap';
        d.style.wordWrap = 'break-word';
        d.style.height = 'auto';
        const h = d.scrollHeight;
        if (h && h > 0) d.style.height = h + 'px';
      });
    }

    document.addEventListener('DOMContentLoaded', () => autosizeRightPanelTextareasAndDivs());
    const observer = new MutationObserver(() => autosizeRightPanelTextareasAndDivs());
    const root = document.querySelector('body');
    if (root) observer.observe(root, { childList: true, subtree: true });
    window.addEventListener('resize', () => autosizeRightPanelTextareasAndDivs());
    </script>
    """,
    visible=False
)


focus_js = gr.HTML(
    """
    <script>
    document.addEventListener('click', function(e) {
      try {
        const el = e.target;
        if (!el) return;
        if ((el.tagName === 'BUTTON' || el.getAttribute('role') === 'button') && el.innerText && el.innerText.trim().toLowerCase().includes('send')) {
          setTimeout(() => {
            let ta = document.querySelector('textarea[placeholder*="Type"]') || document.querySelector('textarea') || document.querySelector('input[type="text"]');
            if (ta) {
              ta.focus();
              const len = ta.value ? ta.value.length : 0;
              if (ta.setSelectionRange) ta.setSelectionRange(len, len);
            }
          }, 120);
        }
      } catch(err) { /* ignore */ }
    }, true);
    </script>
    """,
    visible=False
)

# ----------------- Build Gradio UI -----------------
with gr.Blocks(title="Chatbot + Mood Analytics") as demo:
    gr.Markdown("## Chatbot with Per-Message Sentiment and Mood Trend Analytics")
    demo_css_js
    focus_js

    with gr.Row():
        # Left column: Chat area
        with gr.Column(scale=3):
            backend_url = gr.Textbox(label="Backend URL", value=DEFAULT_BACKEND_URL)
            username = gr.Textbox(label="Username", placeholder="Enter a username")
            login_btn = gr.Button("Login")
            login_status = gr.Text(label="Login status", interactive=False)

            logged_in_state = gr.State(value=None)

            chatbot = gr.Chatbot(label="Conversation", height=480)
            message_box = gr.Textbox(label="Type message", placeholder="Type message", lines=2)
            send_btn = gr.Button("Send")
            load_btn = gr.Button("Load history")
            clear_btn = gr.Button("Clear chat (UI only)")

            # Wiring: login, load history, send, clear
            login_btn.click(fn=login, inputs=[backend_url, username], outputs=[login_status, logged_in_state])
            # When logged_in_state changes, load history into chatbot and sentiment box
            logged_in_state.change(fn=load_history, inputs=[backend_url, logged_in_state], outputs=[chatbot, gr.Textbox(visible=False)])
            # We'll also wire the Load button to refresh both history and sentiment; later we replace the second output with sentiment_box after it's declared.
            load_btn.click(fn=load_history, inputs=[backend_url, logged_in_state], outputs=[chatbot, gr.Textbox(visible=False)])
            # send_button and enter submit will be wired after sentiment_box is defined (below)
            clear_btn.click(fn=clear_chat, inputs=None, outputs=[chatbot, login_status])

        # Right column: analytics + mood trend (dynamic)
        with gr.Column(scale=1, elem_classes="right-panel"):
            gr.Markdown("### Conversation sentiment & Mood Trend")
            sentiment_box = gr.Textbox(
                label="Conversation sentiment (label + score)",
                interactive=False,
                value="Not logged in",
                elem_classes="sentiment-box",
                lines=2
            )
            summary_box = gr.Textbox(
                label="Mood summary (short)",
                interactive=False,
                value="No data",
                elem_classes="summary-box",
                lines=6
            )

            mood_image = gr.Image(label="Mood trend", type="pil", interactive=False, elem_classes="mood-image")

            show_trend_btn = gr.Button("Show Mood Trend")
            refresh_btn = gr.Button("Refresh sentiment")

            # Wire the Show Trend button to fetch and display image + summary + sentiment
            show_trend_btn.click(fn=get_and_plot_mood, inputs=[backend_url, logged_in_state], outputs=[mood_image, summary_box, sentiment_box])

            # Safe refresh wiring
            refresh_btn.click(fn=refresh_sentiment_button, inputs=[backend_url, logged_in_state], outputs=[sentiment_box])

            # Interval auto-update if available
            try:
                Interval = gr.Interval  # type: ignore
                interval = Interval(interval=6.0, run_on_load=False)
                interval.check(fn=get_and_plot_mood, inputs=[backend_url, logged_in_state], outputs=[mood_image, summary_box, sentiment_box])
            except Exception:
                pass

            gr.Markdown(
                "- Click **Show Mood Trend** to view polarity over time and a short summary.\n"
                "- Use **Refresh** to update the conversation sentiment label only.\n"
                "- Auto-refresh will run every 6s if supported."
            )

    # --- Re-wire load & send now that sentiment_box exists ---
    # Replace earlier outputs to point to sentiment_box instead of a dummy textbox
    # (Gradio doesn't support reassigning handlers easily; we define duplicate proper handlers)
    load_btn.click(fn=load_history, inputs=[backend_url, logged_in_state], outputs=[chatbot, sentiment_box])
    logged_in_state.change(fn=load_history, inputs=[backend_url, logged_in_state], outputs=[chatbot, sentiment_box])

    # Send wiring: returns (chat_history, sentiment_text, cleared input)
    send_btn.click(fn=send_message, inputs=[backend_url, logged_in_state, message_box], outputs=[chatbot, sentiment_box, message_box])
    message_box.submit(fn=send_message, inputs=[backend_url, logged_in_state, message_box], outputs=[chatbot, sentiment_box, message_box])

    # Also wire show_trend to auto-refresh sentiment_box (already done above)
    # End of UI building

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
