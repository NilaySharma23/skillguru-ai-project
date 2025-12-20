import requests
import json
import time
import traceback
import os

from rag_adapter import KritikaRAGAdapter  

# ---------------- CONFIG ----------------
API_KEY = "api_key"
MODEL = "gemini-2.5-flash"   
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

KEEP_HISTORY = 12
CONSOLIDATE_AFTER = 10
TIMEOUT = 30


# ------------- GEMINI CALL -------------
def call_gemini(prompt, generation_config=None):

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }

    if generation_config:
        payload["generationConfig"] = generation_config

    try:
        r = requests.post(BASE_URL, json=payload, timeout=TIMEOUT)
        data = r.json()

        if r.status_code != 200:
            return {"ok": False, "error": data}

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"ok": True, "text": text}

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---------------- ULTRA TUTOR ----------------
class UltraTutor:
    def __init__(self):
        self.memory = {
            "history": [],
            "topics_seen": [],
            "weakness_scores": {},
            "difficulty": {},
            "interaction_count": 0,
            "consolidation_summary": None
        }

        #  RAG ADAPTER
        self.rag = KritikaRAGAdapter(enabled=True)

    # -------- Memory helpers --------
    def push_turn(self, user, tutor):
        self.memory["history"].append({"user": user, "tutor": tutor})
        self.memory["history"] = self.memory["history"][-KEEP_HISTORY:]

    def register_topic(self, topic):
        if not topic:
            return
        if topic not in self.memory["topics_seen"]:
            self.memory["topics_seen"].append(topic)
            self.memory["weakness_scores"][topic] = 0
            self.memory["difficulty"][topic] = "intermediate"

    def add_weakness(self, topic, signal):
        if not topic:
            topic = "general"
        self.memory["weakness_scores"].setdefault(topic, 0)

        if signal == "mistake":
            self.memory["weakness_scores"][topic] += 3
        elif signal == "confusion":
            self.memory["weakness_scores"][topic] += 2
        elif signal == "repeat":
            self.memory["weakness_scores"][topic] += 1

    def recompute_difficulty(self):
        for topic, score in self.memory["weakness_scores"].items():
            if score >= 8:
                self.memory["difficulty"][topic] = "beginner"
            elif score >= 4:
                self.memory["difficulty"][topic] = "intermediate"
            else:
                self.memory["difficulty"][topic] = "advanced"

    # -------- RAG QUERY BUILDER --------
    def build_rag_query(self, user_msg, topic):
        weak_topics = sorted(
            self.memory["weakness_scores"],
            key=lambda x: self.memory["weakness_scores"][x],
            reverse=True
        )[:2]

        parts = [
            f"User question: {user_msg}",
            f"Main topic: {topic}"
        ]

        if weak_topics:
            parts.append(f"Weak topics: {', '.join(weak_topics)}")

        parts.append("Explain clearly with examples.")

        return "\n".join(parts)

    # -------- Prompt --------
    def build_prompt(self, user_msg, rag_context):

        history = "\n".join(
            [f"Student: {t['user']}\nTutor: {t['tutor']}" for t in self.memory["history"]]
        ) or "(no history)"

        rag_block = (
            f"""
GROUNDING CONTEXT (trusted source):
{rag_context}

Rules:
- Prefer this context over prior knowledge
- Do NOT invent facts beyond it
- If context is insufficient, say so clearly
"""
            if rag_context
            else "No external grounding context available."
        )

        return f"""
You are **UltraTutor (RAG-enhanced)**.

You are an intelligent AI tutor who:
- Decides automatically: hint / explanation / deep explanation
- Adapts to student difficulty
- Uses RAG context when provided
- Falls back to own reasoning if RAG is empty

SESSION MEMORY:
weakness_scores = {json.dumps(self.memory['weakness_scores'])}
difficulty = {json.dumps(self.memory['difficulty'])}
topics_seen = {self.memory['topics_seen']}

{rag_block}

Conversation history:
{history}

Student message:
\"\"\"{user_msg}\"\"\"

OUTPUT FORMAT:

1) Tutor response

2) ###META###
TOPIC: <topic_or_unknown>
WEAKNESS_SIGNAL: <none | mistake | confusion | repeat>
DIFFICULTY: <beginner | intermediate | advanced>
ASK_CONSOLIDATION: <yes | no>
"""

    # -------- Main answer --------
    def answer(self, user_msg):

        self.memory["interaction_count"] += 1

        # Step 1: initial prompt (no RAG yet)
        topic_guess = user_msg
        weakness_score = self.memory["weakness_scores"].get(topic_guess, 0)
        difficulty = self.memory["difficulty"].get(topic_guess, "intermediate")

        rag_context = ""
        if self.rag.should_use_rag(weakness_score, difficulty):
            rag_query = self.build_rag_query(user_msg, topic_guess)
            rag_context = self.rag.retrieve(rag_query)

        prompt = self.build_prompt(user_msg, rag_context)

        resp = call_gemini(
            prompt,
            generation_config={
                "temperature": 0.15,
                "maxOutputTokens": 700
            }
        )

        if not resp["ok"]:
            return f" API Error:\n{resp['error']}"

        raw = resp["text"]

        if "###META###" in raw:
            reply, meta = raw.split("###META###", 1)
        else:
            reply, meta = raw, ""

        reply = reply.strip()

        topic = "general"
        weakness = "none"
        difficulty = None
        ask_consolidate = False

        for line in meta.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip().upper(), v.strip()
            if k == "TOPIC":
                topic = v if v.lower() != "unknown" else "general"
            elif k == "WEAKNESS_SIGNAL":
                weakness = v
            elif k == "DIFFICULTY":
                difficulty = v
            elif k == "ASK_CONSOLIDATION":
                ask_consolidate = v.lower() == "yes"

        self.register_topic(topic)

        if weakness != "none":
            self.add_weakness(topic, weakness)

        if difficulty:
            self.memory["difficulty"][topic] = difficulty

        if ask_consolidate or self.memory["interaction_count"] % CONSOLIDATE_AFTER == 0:
            pass  # consolidation unchanged

        self.push_turn(user_msg, reply)
        self.recompute_difficulty()

        return reply


# ---------------- CLI ----------------
if __name__ == "__main__":
    print("\n🎓 Intelligent AI Tutor (RAG-Enhanced)")
    tutor = UltraTutor()

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Tutor: Goodbye! 👋")
            break

        ans = tutor.answer(user)
        print("\nTutor:", ans, "\n")
