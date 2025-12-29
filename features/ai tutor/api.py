# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from tutor import UltraTutorV6
from models import ChatRequest, ChatResponse

# ---------------- APP ----------------
app = FastAPI(title="AI Tutor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SESSION STORE ----------------
# session_id -> UltraTutorV6 instance
SESSIONS = {}


def get_tutor(session_id: str) -> UltraTutorV6:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = UltraTutorV6()
    return SESSIONS[session_id]


# ---------------- ROUTES ----------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        tutor = get_tutor(req.session_id)
        reply = tutor.handle_user(req.message)
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset_session(session_id: str):
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"status": "reset successful"}
