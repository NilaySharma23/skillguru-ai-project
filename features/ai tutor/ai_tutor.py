from rag_adapter import KritikaRAGAdapter

import requests
import json
import traceback
import time
from typing import Optional

from dotenv import load_dotenv
load_dotenv()
import os
# ---------------- CONFIG ----------------
API_KEY = os.getenv('API_KEY')         
MODEL = "gemini-2.5-flash"             
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

# Fine-tuneable parameters
KEEP_HISTORY = 12                      # number of last turns to keep in immediate context
CONSOLIDATE_AFTER = 10                 # run consolidation every N user messages
TIMEOUT_SECONDS = 25                   # request timeout

# ------------------- UTIL -------------------
def call_gemini_rest(prompt: str, timeout: int = TIMEOUT_SECONDS) -> str:
    """
    Calls Gemini REST API and returns text (best-effort).
    """
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    try:
        r = requests.post(BASE_URL, json=payload, timeout=timeout)
        data = r.json()
        if "error" in data:
            return f"⚠️ GEMINI ERROR: {data['error'].get('message', 'unknown')}"
        # typical response shape: candidates -> content -> parts -> text
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        traceback.print_exc()
        return "⚠️ Error contacting Gemini REST API (see server logs)."

# ------------------- ULTRATUTOR CLASS -------------------
class UltraTutorV6:
    def __init__(self):
        # Core session memory
        self.memory = {
            "history": [],              # list of {"user":..., "tutor":..., "ts":...}
            "topics_seen": [],          # list of topic strings
            "weakness_scores": {},      # topic -> numeric score (higher => weaker)
            "difficulty": {},           # topic -> beginner/intermediate/advanced
            "interaction_count": 0,     # number of user messages in this session
            "last_consolidation_ts": None,
            "consolidation_summary": None,   # stores results of last consolidation
            "rag_chunks": [],           # placeholder for RAG texts
        }
        # runtime flags
        self.quiz_active = False
        self.last_quiz = None  # {"question":..., "answer":..., "topic":...}
        self.rag = KritikaRAGAdapter(enabled=True)


    # ---------- Memory helpers ----------
    def push_turn(self, user_text: str, tutor_text: str):
        turn = {"user": user_text, "tutor": tutor_text, "ts": time.time()}
        self.memory["history"].append(turn)
        self.memory["history"] = self.memory["history"][-KEEP_HISTORY:]

    def register_topic(self, topic: str):
        if not topic:
            return
        if topic not in self.memory["topics_seen"]:
            self.memory["topics_seen"].append(topic)
            # initialize scores
            self.memory["weakness_scores"].setdefault(topic, 0)
            self.memory["difficulty"].setdefault(topic, "intermediate")
            
    def _update_rag_chunks(self, user_msg: str, topic: str):
        """
        Passive RAG enrichment.
        Does NOT affect tutor behavior.
        """
        query = f"Question: {user_msg}\nTopic: {topic}"
        chunks = self.rag.retrieve(query, top_k=3)
        if chunks:
            self.memory["rag_chunks"] = chunks

    # ---------- Weakness signals ----------
    def add_weakness_signal(self, topic: str, signal: str):
        """
        signal: one of 'mistake', 'confusion', 'repeat'
        Weighting:
          - mistake: +3
          - confusion: +2
          - repeat: +1
        """
        if not topic:
            topic = "general"
        self.register_topic(topic)
        self._update_rag_chunks(user_msg, topic)
        if signal == "mistake":
            self.memory["weakness_scores"][topic] += 3
        elif signal == "confusion":
            self.memory["weakness_scores"][topic] += 2
        elif signal == "repeat":
            self.memory["weakness_scores"][topic] += 1

    def get_weakest_topics(self, top_k: int = 5):
        scores = self.memory["weakness_scores"]
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    # ---------- Difficulty recommender ----------
    def recompute_difficulty_local(self):
        """
        Heuristic mapping from weakness score to difficulty recommendation:
        high weakness => beginner, medium => intermediate, low => advanced.
        """
        for topic, score in self.memory["weakness_scores"].items():
            if score >= 8:
                level = "beginner"
            elif score >= 4:
                level = "intermediate"
            else:
                level = "advanced"
            self.memory["difficulty"][topic] = level

    # ---------- RAG placeholder ----------
    def rag_snippet(self) -> str:
        if not self.memory["rag_chunks"]:
            return "(no external knowledge loaded)"
        return "\n\n".join(self.memory["rag_chunks"][-3:])

    # ---------- Consolidation (heavy analysis run) ----------
    def run_consolidation(self) -> dict:
        """
        Calls the LLM once to analyze the session so far and produce:
         - ranked weaknesses & reasons
         - suggested difficulty per topic
         - career mapping suggestions (roles, skills, learning steps, timeline)
        This is called automatically after every CONSOLIDATE_AFTER interactions,
        or can be triggered manually via /career.
        """
        history_text = "\n".join(
            [f"Student: {t['user']}\nTutor: {t['tutor']}" for t in self.memory["history"][-100:]]
        ) or "(no history)"
        rag_text = self.rag_snippet()
        prompt = f"""
You are an expert AI tutoring analyst. Analyze the student's session and provide a JSON block with:
1) ranked_weaknesses: list of objects {{topic, score, short_reason}}
2) difficulty_recs: map topic -> one of beginner/intermediate/advanced
3) quick_strengths: short list of topics student seems strong in
4) career_mappings: list of up to 5 recommended career paths based on student's strengths + weaknesses.
  For each career provide: name, primary_skills (list), secondary_skills (list), short_roadmap (3-6 bullets), estimated_timeline (months)
5) suggested_next_actions: up to 6 concrete next steps (practice, readings, quizzes)

Session history:
{history_text}

RAG knowledge (if any):
{rag_text}

Return ONLY valid JSON. Example:
{{
  "ranked_weaknesses": [{{"topic":"X","score":9,"short_reason":"keeps asking for repeats"}}],
  "difficulty_recs": {{"X":"beginner"}},
  "quick_strengths": ["A","B"],
  "career_mappings": [{{"name":"ML Engineer","primary_skills":["python","ml"],"secondary_skills":["devops"],"short_roadmap":["..."],"estimated_timeline":"6-12 months"}}],
  "suggested_next_actions": ["..."]
}}
"""
        raw = call_gemini_rest(prompt)
        # Try to parse JSON from response
        parsed = {}
        try:
            # Attempt to locate a JSON substring:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            jtxt = raw[start:end]
            parsed = json.loads(jtxt)
        except Exception:
            # fallback heuristic: return minimal report
            parsed = {
                "ranked_weaknesses": [{"topic": t, "score": s, "short_reason": "auto-detected"} for t, s in self.get_weakest_topics()],
                "difficulty_recs": self.memory.get("difficulty", {}),
                "quick_strengths": [],
                "career_mappings": [],
                "suggested_next_actions": ["Try a short quiz on your weakest topic.", "Review fundamentals and examples."],
            }
        # Store consolidation
        self.memory["consolidation_summary"] = parsed
        self.memory["last_consolidation_ts"] = time.time()
        # Update local weakness + difficulty from parsed results if present
        for w in parsed.get("ranked_weaknesses", []):
            t = w.get("topic")
            s = int(w.get("score", 0))
            if t:
                # set to max of existing and parsed score
                current = self.memory["weakness_scores"].get(t, 0)
                self.memory["weakness_scores"][t] = max(current, s)
        for t, lvl in parsed.get("difficulty_recs", {}).items():
            self.memory["difficulty"][t] = lvl
        return parsed

    # ---------- Build interactive tutor prompt (single-call) ----------
    def build_tutor_prompt(self, user_msg: str) -> str:
        hist = "\n".join([f"Student: {t['user']}\nTutor: {t['tutor']}" for t in self.memory["history"][-KEEP_HISTORY:]]) or "(no history)"
        # compute suggested difficulty locally as well
        self.recompute_difficulty_local()
        difficulty_summary = json.dumps(self.memory["difficulty"], indent=2)
        weakness_summary = json.dumps(self.memory["weakness_scores"], indent=2)
        prompt = f"""
You are an expert, empathetic AI Tutor. Use the session memory below to:
- Decide if the student needs a HINT, EXPLANATION, or DEEP EXPLANATION.
- Use difficulty recommendations (beginner/intermediate/advanced) to choose tone & depth.
- Update weakness signals if the student shows confusion/mistakes/repeats.
- If last tutor message was a quiz question, grade the student's answer and mark weakness signals appropriately.
- Be supportive and give concrete next steps.

SESSION MEMORY (JSON summary):
{{
  "weakness_scores": {weakness_summary},
  "difficulty": {difficulty_summary},
  "topics_seen": {self.memory["topics_seen"]},
  "interaction_count": {self.memory["interaction_count"]}
}}

Recent conversation:
{hist}

RAG_SNIPPET:
{self.rag_snippet()}

Student message:
\"\"\"{user_msg}\"\"\"

OUTPUT required (two parts):
1) Tutor reply to show the student.
2) A META block that begins with EXACTLY "###META###" and contains lines:
TOPIC: <topic-or-'unknown'>
WEAKNESS_SIGNAL: <none|mistake|confusion|repeat>
DIFFICULTY: <beginner|intermediate|advanced>
QUIZ_FEEDBACK: <none|correct|partially_correct|incorrect>
EXPLICIT_ASK_CONSOLIDATION: <yes|no>   # yes if you (LLM) think we should run consolidation now

The student will see only the tutor reply (part 1). The META block is for the system to parse.
"""
        return prompt

    # ---------- Decide weakness signal from meta line ----------
    def parse_meta_block(self, meta_text: str) -> dict:
        res = {"topic": None, "weakness_signal": "none", "difficulty": None, "quiz_feedback": "none", "ask_consolidation": False}
        for line in meta_text.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip().upper()
            v = v.strip()
            if k == "TOPIC":
                res["topic"] = v if v.lower() != "unknown" else None
            elif k == "WEAKNESS_SIGNAL":
                res["weakness_signal"] = v.lower()
            elif k == "DIFFICULTY":
                res["difficulty"] = v.lower()
            elif k == "QUIZ_FEEDBACK":
                res["quiz_feedback"] = v.lower()
            elif k == "EXPLICIT_ASK_CONSOLIDATION":
                res["ask_consolidation"] = ("yes" in v.lower())
        return res

    # ---------- Main interaction entry ----------
    def handle_user(self, user_msg: str) -> str:
        # manual commands
        low = user_msg.strip().lower()
        if low in {"/reset", "reset session"}:
            self.__init__()
            return "Session reset. Ready for a new topic — what would you like to learn?"
        if low in {"/career", "career"}:
            # if we have consolidation summary, return it; otherwise run consolidation
            if self.memory.get("consolidation_summary"):
                return json.dumps(self.memory["consolidation_summary"], indent=2)
            else:
                parsed = self.run_consolidation()
                return json.dumps(parsed, indent=2)

        # increment interaction count
        self.memory["interaction_count"] += 1

        # Build prompt and call LLM once
        prompt = self.build_tutor_prompt(user_msg)
        raw = call_gemini_rest(prompt)
        # default reply empty if error
        reply_text = raw
        meta_text = ""
        if "###META###" in raw:
            reply_text, meta_text = raw.split("###META###", 1)
            reply_text = reply_text.strip()
            meta_text = meta_text.strip()

        meta = self.parse_meta_block(meta_text)
        topic = meta.get("topic") or "general"
        # register topic
        self.register_topic(topic)

        # update weakness signals and difficulty locally
        sig = meta.get("weakness_signal", "none")
        if sig != "none":
            self.add_weakness_signal(topic, sig)
        # update difficulty if provided
        if meta.get("difficulty"):
            self.memory["difficulty"][topic] = meta.get("difficulty")

        # quiz feedback handling
        qfb = meta.get("quiz_feedback")
        if qfb in {"incorrect", "partially_correct", "correct"}:
            # map feedback to weakness signal
            if qfb == "incorrect":
                self.add_weakness_signal(topic, "mistake")
            elif qfb == "partially_correct":
                self.add_weakness_signal(topic, "confusion")
            # store last quiz result if we want
            self.last_quiz = {"topic": topic, "result": qfb, "ts": time.time()}

        # push turn into history
        self.push_turn(user_msg, reply_text)

        # Auto-consolidation decision: either explicit ask or periodic after N interactions
        consolidated = None
        if meta.get("ask_consolidation") or (self.memory["interaction_count"] % CONSOLIDATE_AFTER == 0):
            consolidated = self.run_consolidation()

        # After consolidation, recompute difficulty locally to reflect changes
        self.recompute_difficulty_local()

        # For debugging/visibility we could append a short system note (not shown to user)
        return reply_text

# ------------------- CLI -------------------
if __name__ == "__main__":
    print("\n🎓 AI Tutor")
    print("Type 'exit' to quit.\n")

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
        out = tutor.handle_user(msg)
        print("\nTutor:", out, "\n")
