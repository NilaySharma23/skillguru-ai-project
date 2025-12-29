from rag_adapter import KritikaRAGAdapter
from rag import ingest_all_pdfs_from_dir

import requests
import json
import traceback
import time
import os
from dotenv import load_dotenv
load_dotenv()

# CONFIG 
API_KEY = os.getenv("API_KEY")
MODEL = "gemini-2.5-flash"
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

KEEP_HISTORY = 12
CONSOLIDATE_AFTER = 10
TIMEOUT_SECONDS = 25


#  UTIL 
def call_gemini_rest(prompt: str, timeout: int = TIMEOUT_SECONDS) -> str:
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    try:
        r = requests.post(BASE_URL, json=payload, timeout=timeout)
        data = r.json()
        if "error" in data:
            return f"⚠️ GEMINI ERROR: {data['error'].get('message', 'unknown')}"
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        traceback.print_exc()
        return "⚠️ Error contacting Gemini."


# ULTRA TUTOR
class UltraTutorV6:
    def __init__(self):
        self.memory = {
            "history": [],
            "topics_seen": [],
            "weakness_scores": {},
            "difficulty": {},
            "interaction_count": 0,
            "rag_chunks": [],
            "session_summary": None,
        }

        self.analytics = {
            "confusion_points": {},
            "retries": {},
            "mastery_delta": {},
        }

        self.rag = KritikaRAGAdapter(enabled=True)

    # Persona
    def compute_persona(self) -> str:
        if not self.memory["weakness_scores"]:
            return "intermediate"
        avg = sum(self.memory["weakness_scores"].values()) / max(1, len(self.memory["weakness_scores"]))
        if avg >= 6:
            return "beginner"
        elif avg >= 3:
            return "intermediate"
        return "advanced"

    #  Memory 
    def push_turn(self, user, tutor):
        self.memory["history"].append({"user": user, "tutor": tutor, "ts": time.time()})
        self.memory["history"] = self.memory["history"][-KEEP_HISTORY:]

    def register_topic(self, topic):
        if topic and topic not in self.memory["topics_seen"]:
            self.memory["topics_seen"].append(topic)
            self.memory["weakness_scores"].setdefault(topic, 0)
            self.memory["difficulty"].setdefault(topic, "intermediate")

    # RAG 
    def _update_rag_chunks(self, user_msg, topic):
        query = f"Question: {user_msg}\nTopic: {topic}"
        chunks = self.rag.retrieve(query, top_k=3)
        if chunks:
            self.memory["rag_chunks"] = chunks

    def rag_snippet(self):
        return "\n\n".join(self.memory["rag_chunks"][-3:]) if self.memory["rag_chunks"] else "(no external knowledge)"

    #  Weakness & analytics
    def add_weakness_signal(self, topic, signal):
        self.register_topic(topic)
        self.analytics["confusion_points"].setdefault(topic, 0)
        self.analytics["retries"].setdefault(topic, 0)

        if signal == "mistake":
            self.memory["weakness_scores"][topic] += 3
            self.analytics["confusion_points"][topic] += 1
        elif signal == "confusion":
            self.memory["weakness_scores"][topic] += 2
            self.analytics["confusion_points"][topic] += 1
        elif signal == "repeat":
            self.memory["weakness_scores"][topic] += 1
            self.analytics["retries"][topic] += 1

    #  Summarization 
    def update_session_summary(self):
        old_turns = self.memory["history"][:-KEEP_HISTORY // 2]
        if not old_turns:
            return
        text = "\n".join(f"Student: {t['user']}\nTutor: {t['tutor']}" for t in old_turns)
        prompt = f"""
Summarize this tutoring session briefly.
Focus on:
- topics covered
- confusion areas
- learning progress

{text}
"""
        self.memory["session_summary"] = call_gemini_rest(prompt).strip()

    #  Prompt 
    def build_prompt(self, user_msg):
        persona = self.compute_persona()

        recent_hist = "\n".join(
            f"Student: {t['user']}\nTutor: {t['tutor']}"
            for t in self.memory["history"][-KEEP_HISTORY:]
        )

        last_topic = (
            self.memory["topics_seen"][-1]
            if self.memory["topics_seen"]
            else "the current topic"
        )

        return f"""
You are a real, advanced AI tutor.

STUDENT PERSONA: {persona}
(adapt tone, pacing, and depth accordingly)

SESSION SUMMARY:
{self.memory["session_summary"] or "(none yet)"}

RECENT CONVERSATION:
{recent_hist}

RAG CONTEXT (authoritative reference if relevant):
- If the answer is clearly present or supported here, use it.
- If not relevant or insufficient, answer normally using your own knowledge.
{self.rag_snippet()}

CURRENT TOPIC (IMPORTANT — DO NOT CHANGE):
{last_topic}

Student message:
\"\"\"{user_msg}\"\"\"

Respond naturally like a calm, confident human tutor.
Then include a META block.

###META###
TOPIC: <topic-or-unknown>
WEAKNESS_SIGNAL: <none|mistake|confusion|repeat>
DIFFICULTY: <beginner|intermediate|advanced>

IMPORTANT BEHAVIOR RULES (CHATGPT-LIKE):

1. Topic continuity (DEFAULT):
   - Assume the student wants to continue with the current topic.
   - Short or vague messages (e.g., "not understand", "confused", "what?")
     are ALWAYS follow-ups to the current topic.
   - In these cases, re-explain or simplify the SAME topic.

2. Topic change (ONLY when clearly introduced):
   - If the student clearly asks about a new, unrelated subject
     (e.g., a different field, person, place, or concept),
     IMMEDIATELY switch to the new topic.
   - Do NOT mention the previous topic.
   - Do NOT ask for confirmation.
   - Treat it as a fresh question.

3. Teaching style:
   - Do NOT ask diagnostic questions unless explicitly requested.
   - Prefer direct, clear explanations.
   - Adjust depth naturally based on student understanding.

4. Natural behavior:
   - Behave like ChatGPT or a skilled human tutor.
   - Be flexible, calm, and student-driven.
   - Follow the student’s intent, not rigid rules.
"""


    # MAIN
    def handle_user(self, user_msg):
        self.memory["interaction_count"] += 1

        prompt = self.build_prompt(user_msg)
        raw = call_gemini_rest(prompt)

        reply, meta = raw, ""
        if "###META###" in raw:
            reply, meta = raw.split("###META###", 1)
            reply = reply.strip()

        topic, signal = "general", "none"
        for line in meta.splitlines():
            if line.startswith("TOPIC"):
                topic = line.split(":")[1].strip()
            elif line.startswith("WEAKNESS_SIGNAL"):
                signal = line.split(":")[1].strip()

        self.register_topic(topic)
        self._update_rag_chunks(user_msg, topic)

        if signal != "none":
            self.add_weakness_signal(topic, signal)

        self.push_turn(user_msg, reply)

        if self.memory["interaction_count"] % CONSOLIDATE_AFTER == 0:
            self.update_session_summary()

        return reply


# CLI
import sys

def stream_print(text: str, delay: float = 0.03):
    """
    Prints text word-by-word like GPT's.
    """
    words = text.split(" ")
    for i, word in enumerate(words):
        sys.stdout.write(word)
        if i < len(words) - 1:
            sys.stdout.write(" ")
        sys.stdout.flush()
        time.sleep(delay)
    print()  # newline at end

if __name__ == "__main__":
    ingest_all_pdfs_from_dir("data/pdfs")

    print("\n🎓 Intelligent AI Tutor (Advanced)")
    tutor = UltraTutorV6()

    while True:
        try:
            msg = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTutor: Bye! Keep learning 🚀")
            break
        if msg.lower() in {"exit", "quit"}:
            print("Tutor: Bye! Keep learning 🚀")
            break
        reply = tutor.handle_user(msg)
        print("\nTutor: ", end="", flush=True)
        stream_print(reply, delay=0.03)
        print()

