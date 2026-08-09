"""
main.py

FastAPI application exposing the NETSOL chatbot as a web API.

Run with:
    uvicorn main:api

Then test with:
    POST http://127.0.0.1:8000/chat
    body: {"question": "What awards has NETSOL won?"}

Or view interactive docs at:
    http://127.0.0.1:8000/docs
"""

from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import document_store
import token_tracker
from chatbot import ask_detailed, check_google_status, reset_history


api = FastAPI(title="NETSOL Chatbot API")

# Allow the frontend (opened as a local HTML file, or served from any origin)
# to call this API from the browser. For a real production deployment, you'd
# restrict allow_origins to your actual frontend's domain instead of "*".
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # set when the frontend has an uploaded document


class UsageInfo(BaseModel):
    request_number: Optional[int] = None
    model: Optional[str] = None
    llm_calls: int = 0
    response_time_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    running_total_tokens: int = 0
    running_total_cost: float = 0.0


class ChatResponse(BaseModel):
    question: str
    answer: str
    route: str
    cached: bool
    usage: Optional[UsageInfo] = None


class UploadResponse(BaseModel):
    kind: str
    filename: str
    chars: Optional[int] = None
    rows: Optional[int] = None
    columns: Optional[int] = None


@api.get("/")
def root():
    return {"status": "ok", "message": "NETSOL chatbot API is running. POST to /chat with a question."}


@api.get("/status")
def status():
    """Checks whether Google is reporting any active Gemini/Vertex AI incidents —
    useful when you're seeing repeated 503 errors and want to know if it's a
    known, broader outage rather than something local."""
    return check_google_status()


@api.get("/usage")
def usage():
    """Cumulative token usage and estimated cost since this server process
    started, independent of any single request."""
    return token_tracker.cumulative()


@api.post("/upload", response_model=UploadResponse)
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)):
    """Extracts/parses an uploaded file and stores it in memory for this
    session only (see document_store.py). A .csv is parsed into a
    dataframe and becomes the primary source for all following questions
    in this session; .txt/.pdf are stored as extracted text. Overwrites
    anything previously uploaded by the same session."""
    file_bytes = await file.read()
    try:
        result = document_store.store_document(session_id, file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    reset_history(session_id)
    return UploadResponse(**result)


@api.post("/clear-document")
def clear_document(session_id: str = Form(...)):
    document_store.clear_document(session_id)
    reset_history(session_id)
    return {"status": "ok"}


@api.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    with token_tracker.track_request() as tracked:
        try:
            result = ask_detailed(request.question, session_id=request.session_id)
            answer = result["answer"]
            route = result["route"]
            cached = result["cached"]
        except Exception as e:
            answer = f"Something went wrong answering that: {e}"
            route = "ERROR"
            cached = False
        usage = tracked.summary()

    return ChatResponse(question=request.question, answer=answer, route=route, cached=cached, usage=usage)

