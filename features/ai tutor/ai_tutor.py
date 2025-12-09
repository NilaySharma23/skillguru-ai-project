import requests
import json
import traceback


API_KEY = "api key"
MODEL = "gemini-2.5-flash"  
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"


class UltraTutorV4:
    """
    UltraTutor v4:
    - Dynamic weakness profiling
    - Difficulty-aware explanations
    - RAG-ready structure
    - Quiz-aware adaptive logic
    - Single Gemini call per message
    """

    def __init__(self):
        self.memory = {
            "conversation": [],
            "topics": [],
            "weakness_scores": {},      # topic: numeric value
            "difficulty_profile": {},   # topic: beginner/intermediate/advanced
            "quiz_active": False,
            "last_quiz": None,
            "rag_chunks": [],           # RAG texts (future expansion)
        }

    # ------------- WEAKNESS PROFILING --------------- #

    def update_weakness(self, topic, signal):
        """
        signal may be:
        - 'mistake'
        - 'confusion'
        - 'repeat_request'
        """
        if topic not in self.memory["weakness_scores"]:
            self.memory["weakness_scores"][topic] = 0

        if signal == "mistake":
            self.memory["weakness_scores"][topic] += 2
        elif signal == "confusion":
            self.memory["weakness_scores"][topic] += 1
        elif signal == "repeat_request":
            self.memory["weakness_scores"][topic] += 1

    def get_weakness_level(self, topic):
        score = self.memory["weakness_scores"].get(topic, 0)
        if score >= 4:
            return "weak"
        if score >= 2:
            return "medium"
        return "strong"

    # ----------- DIFFICULTY RECOMMENDER ------------ #

    def recommend_level(self, topic):
        """
        Uses weakness score + topic history to infer difficulty.
        """
        weakness = self.get_weakness_level(topic)

        if weakness == "weak":
            return "beginner"
        elif weakness == "medium":
            return "intermediate"
        else:
            return "advanced"

    # ---------------- RAG HOOK ---------------- #

    def rag_context(self):
        """
        Combine all stored RAG chunks into a single reference text.
        (Future: integrate FAISS / Chroma vector DB)
        """
        if not self.memory["rag_chunks"]:
            return "No RAG knowledge available."
        return "\n".join(self.memory["rag_chunks"][-3:])  # last 3 chunks max

    # ---------------- GEMINI CALL ---------------- #

    def call_gemini(self, prompt):
        try:
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
            response = requests.post(URL, json=payload).json()

            if "error" in response:
                return f"⚠️ Gemini Error: {response['error']['message']}"

            return response["candidates"][0]["content"]["parts"][0]["text"]

        except Exception:
            traceback.print_exc()
            return "⚠️ Technical issue contacting Gemini."

    # ---------------- PROMPT BUILDER ---------------- #

    def build_prompt(self, user_msg):
        mem_json = json.dumps(self.memory, indent=2)
        rag_text = self.rag_context()
        history = json.dumps(self.memory["conversation"][-12:], indent=2)

        return f"""
You are **SkillGuru UltraTutor v4**, an advanced AI tutor.

You have MEMORY about the learner:
{mem_json}

You also have optional external RAG knowledge:
<<RAG_KNOWLEDGE>>
{rag_text}
<<END_RAG>>

Your responsibilities:
- Automatically infer student's difficulty level.
- Detect confusion, weakness, repeated questions.
- Adapt teaching style dynamically.
- Choose internally:
  HINT / EXPLANATION / DEEP EXPLANATION / QUIZ_QUESTION /
  QUIZ_FEEDBACK / SUMMARY / SMALLTALK / MOTIVATION
- Never ask the learner what teaching mode they want.
- Use weakness profiling + difficulty recommendation.
- If quiz is active, interpret message as student's answer.
- Automatically infer topic.

You MUST output TWO PARTS:

(1) Message for the student.
(2) A hidden block:

###META###
TOPIC: <topic>
QUIZ_ACTIVE: <yes/no>
DIFFICULTY: <beginner/intermediate/advanced>
WEAKNESS_SIGNAL: <none/confusion/mistake/repeat_request>

Conversation history:
{history}

Student message:
\"\"\"{user_msg}\"\"\"
""".strip()

    # ---------------- MAIN ---------------- #

    def ask(self, user_msg):

        # Manual reset
        if user_msg.lower() in ["/reset", "reset session"]:
            self.__init__()
            return "Session reset. Let's start fresh!"

        # Build prompt
        prompt = self.build_prompt(user_msg)

        # Call Gemini
        raw = self.call_gemini(prompt)

        # Parse output
        answer, meta = raw, ""
        if "###META###" in raw:
            answer, meta = raw.split("###META###", 1)
            meta = meta.strip()

        topic = None
        quiz_active = False
        diff = "beginner"
        weakness_signal = "none"

        for line in meta.splitlines():
            if line.startswith("TOPIC:"):
                topic = line.split(":")[1].strip()
            elif line.startswith("QUIZ_ACTIVE:"):
                quiz_active = "yes" in line.lower()
            elif line.startswith("DIFFICULTY:"):
                diff = line.split(":")[1].strip()
            elif line.startswith("WEAKNESS_SIGNAL:"):
                weakness_signal = line.split(":")[1].strip()

        # Update memory & profiling
        if topic:
            if topic not in self.memory["topics"]:
                self.memory["topics"].append(topic)

        if weakness_signal in ["mistake", "confusion", "repeat_request"]:
            self.update_weakness(topic, weakness_signal)

        self.memory["quiz_active"] = quiz_active
        self.memory["conversation"].append({"user": user_msg, "tutor": answer.strip()})

        return answer.strip()


# ---------------- CLI ---------------- #
if __name__ == "__main__":
    print("\n🎓 SkillGuru AI Tutor")
    print("Type 'exit' to quit | /reset to restart\n")

    tutor = UltraTutorV4()

    while True:
        msg = input("You: ").strip()
        if msg.lower() in ["exit", "quit"]:
            print("Tutor: Goodbye! Keep learning")
            break

        reply = tutor.ask(msg)
        print("\nTutor:", reply, "\n")
