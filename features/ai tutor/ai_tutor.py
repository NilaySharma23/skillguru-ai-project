import requests
import json
import traceback

API_KEY = "apikey_here"
MODEL = "gemini-2.5-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"


class UltraTutor:
    def __init__(self):
        self.memory = {
            "conversation": [],
            "topics": [],
            "progress": {},
            "weaknesses": [],
            "quiz_active": False,
            "last_quiz": None,
        }

    # ---------------- MEMORY UTILITIES ---------------- #

    def add_topic(self, topic):
        if topic and topic not in self.memory["topics"]:
            self.memory["topics"].append(topic)

    def add_progress(self, topic, result):
        """Track student performance per topic"""
        if topic not in self.memory["progress"]:
            self.memory["progress"][topic] = {"correct": 0, "incorrect": 0}

        if result == "correct":
            self.memory["progress"][topic]["correct"] += 1
        else:
            self.memory["progress"][topic]["incorrect"] += 1
            if topic not in self.memory["weaknesses"]:
                self.memory["weaknesses"].append(topic)

    def get_summary_memory(self):
        """Readable memory summary"""
        return json.dumps(self.memory, indent=2)

    # --------------- GEMINI CALL ---------------- #

    def call_gemini(self, prompt):
        try:
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
            response = requests.post(URL, json=payload).json()

            if "error" in response:
                return f"⚠ API Error: {response['error']['message']}"

            return response["candidates"][0]["content"]["parts"][0]["text"]

        except Exception:
            traceback.print_exc()
            return "⚠ Technical issue contacting Gemini."

    # --------------- PROMPT BUILDER ---------------- #

    def build_prompt(self, user_msg):
        memory_json = self.get_summary_memory()

        return f"""
You are **SkillGuru UltraTutor v3**, a highly advanced AI tutor using Gemini 2.5 Flash.

You have MEMORY about the learner:
{memory_json}

Your job:
- Understand the student's intent and difficulty level.
- Decide *internally* the correct teaching strategy:
  HINT / EXPLANATION / DEEP EXPLANATION / QUIZ QUESTION /
  QUIZ FEEDBACK / SUMMARY / SMALLTALK / MOTIVATION.
- Never ask what mode they want.
- Automatically infer the topic, confusion level, and next best step.
- Use past mistakes to adapt teaching.
- If quiz is active, treat student message as an answer and grade it.
- Improve your future teaching using long-term memory.

OUTPUT FORMAT:

1) The tutor's message to the student.

2) A hidden meta block starting EXACTLY with:
###META###
TOPIC: <topic name or 'unknown'>
QUIZ_ACTIVE: <yes/no>
RESULT: <correct/incorrect/none>
        """.strip() + f"""

Conversation history:
{self.memory['conversation']}

Student message:
\"\"\"{user_msg}\"\"\"
"""

    # --------------- MAIN LOGIC ---------------- #

    def ask(self, user_msg):

        # manual reset
        if user_msg.lower() in ["/reset", "reset session"]:
            self.memory = {
                "conversation": [],
                "topics": [],
                "progress": {},
                "weaknesses": [],
                "quiz_active": False,
                "last_quiz": None,
            }
            return "Session reset. What topic can I help you with now? 😊"

        # build prompt
        prompt = self.build_prompt(user_msg)

        # call LLM
        raw = self.call_gemini(prompt)

        # split meta block
        answer, meta = raw, ""
        if "###META###" in raw:
            answer, meta = raw.split("###META###", 1)
            meta = meta.strip()

        topic = None
        quiz_active = False
        result = "none"

        for line in meta.splitlines():
            if line.startswith("TOPIC:"):
                topic = line.split(":")[1].strip()
            elif line.startswith("QUIZ_ACTIVE:"):
                quiz_active = "yes" in line.lower()
            elif line.startswith("RESULT:"):
                result = line.split(":")[1].strip().lower()

        # update memory
        if topic:
            self.add_topic(topic)

        if result in ["correct", "incorrect"] and topic:
            self.add_progress(topic, result)

        self.memory["quiz_active"] = quiz_active
        self.memory["conversation"].append({"user": user_msg, "tutor": answer.strip()})

        return answer.strip()


# ---------------- CLI ---------------- #
if __name__ == "__main__":
    print("\n🎓 SkillGuru UltraTutor v3 — Gemini 2.5 Flash")
    print("Type 'exit' to quit | /reset to restart\n")

    tutor = UltraTutor()

    while True:
        msg = input("You: ").strip()
        if msg.lower() in ["exit", "quit"]:
            print("Tutor: Goodbye! Keep learning 🌟")
            break

        reply = tutor.ask(msg)
        print("\nTutor:", reply, "\n")
