
# ✨ Chatbot with Sentiment & Mood Analytics  


---

## 🚀 Overview  
This project is a **full-stack intelligent chatbot** featuring:  

### ✅ Statement-Level Sentiment  
Every user message is analyzed individually using Transformers.  
Sentiment is displayed **inline** in the chat.  

### ✅ Conversation-Level Sentiment  
A running sentiment score is computed and shown in the right panel.  

### ✅ Mood Trend Visualization (Additional Credit)  
A moving-averaged mood curve visually shows emotional flow.  
Includes:  
- 📉 Shift detection  
- 🧠 Mood summary in plain English  
- 📊 Trend graph with raw + smoothed polarity  

---

# 🛠 Tech Stack  
### **Backend**
- ⚡ FastAPI  
- 🗄 SQLite  
- 🤗 Transformers (HuggingFace)  
- 🧪 Pydantic  
- 📈 Matplotlib  

### **Frontend**
- 🎨 Gradio  
- 🪄 Custom JS for auto-resizing & input autofocus  
- 🧱 Responsive analytics layout  

---

# ▶️ How to Run

### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Start the Backend**
```bash
uvicorn app.main:app --reload
```

### **3️⃣ Launch the Gradio UI**
```bash
python gradio_app.py
```
---

# 🧑‍💼 creating user 
For simplicity purpouse and still able to save user history everything is tied to user ID and to create a new user simply enter the name and press login for the Gradio UI.
From backend there is a route to create user.

---

# 🧩 Sentiment Logic

## 🔹 1. Per-Message Sentiment  
Each user message goes through a HuggingFace sentiment classifier:  
Outputs include:  
- `sentiment_label` (Positive/Negative)  
- `sentiment_score`  
- `polarity` ∈ [-1, +1]  

Displayed like:  
```
I feel good today! — [POSITIVE 0.98]
```

---

## 🔹 2. Conversation-Level Sentiment  
Computed as the **average polarity** of all user messages:  
```
avg_polarity = sum(polarities) / count
```

Displayed as:  
```
Positive (+0.45)
```

---

## 🔹 3. Mood Trend (Additional Credit)  

### 📈 Raw Polarity  
Example:  
```
[0.9, 0.8, 0.1, -0.4, -0.1]
```

### 📉 Smoothed Trend  
Moving average applied to identify direction.  

### ⚠ Mood Shift Detection  
If the smoothed score drops or rises sharply → **“Shift detected”**.

### 🧠 Summary Generation  
Plain-language interpretation like:  
- *Mood improves over time*  
- *Negative dip around message 4*  
- *Highly fluctuating emotional pattern*  

---

# 📊 Status of Tier 2 Requirements

| Feature | Status |
|--------|--------|
| Per-message sentiment | ✅ Completed |
| Display per-message sentiment | ✅ Inline labels |
| Conversation-level sentiment | ✅ Right panel |
| Mood trend graph | ✅ Raw + smoothed |
| Mood shift detection | ✅ Fully implemented |
| Mood summary | ✅ Natural-language summary |
| Fancy UI enhancements | ✅ Dynamic panels, auto-focus, auto-clear |

---

# ✨ Highlights  
- Clean modular architecture  
- Reliable analytics  
- Production-ready UI  
- Flexible for custom LLM integration  

---

NOTE: The DB is created locally during runtime named "chat.db".


