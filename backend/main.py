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

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import backend.document_store as document_store
import backend.services as services
import backend.token_tracker as token_tracker
from backend.chatbot import ask_detailed, check_google_status, reset_history

logger = logging.getLogger("netsol.leads")


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


# ---------- NETSOL services discovery / lead generation ----------
class ServiceInfo(BaseModel):
    id: str
    name: str
    description: str
    category: str
    url: str


class ServiceRequestCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: str
    service_id: str
    description: str = Field(..., min_length=1, max_length=4000)
    phone: Optional[str] = Field(None, max_length=40)
    company: Optional[str] = Field(None, max_length=200)


class ServiceRequestResponse(BaseModel):
    request_id: str
    status: str
    created_at: str
    lead_email_sent: bool
    confirmation_email_sent: bool
    email_warning: Optional[str] = None


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


@api.get("/services", response_model=list[ServiceInfo])
def list_services():
    """The catalog shown in the "Which NETSOL service are you interested
    in?" step — sourced from services.py (NETSOL's real, official
    offerings)."""
    return services.list_services()


@api.post("/service-requests", response_model=ServiceRequestResponse)
def create_service_request(request: ServiceRequestCreate):
    """Creates a new NETSOL service-request lead: validates input, stores
    it (status "New") in netsol_leads.db, then attempts both the internal
    notification email and the customer confirmation email.

    The lead is stored BEFORE any email is attempted, so a failed email
    never loses the lead — see services.py. Any
    email failure is logged and reported back via `email_warning`, but the
    request itself is still a success (the customer sees "Request
    Submitted Successfully")."""
    name = request.name.strip()
    email = request.email.strip()
    description = request.description.strip()

    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")
    if not services.is_valid_email(email):
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    if not description:
        raise HTTPException(status_code=400, detail="Please describe what you're looking for.")

    service = services.get_service(request.service_id)
    if service is None:
        raise HTTPException(status_code=400, detail="Unknown service selected.")

    lead = services.create_lead(
        name=name,
        email=email,
        service_id=service["id"],
        service_name=service["name"],
        description=description,
        phone=request.phone,
        company=request.company,
    )

    warnings = []

    lead_result = services.send_lead_notification_email(lead)
    services.update_email_status(lead["request_id"], lead_email_sent=lead_result.sent)
    if not lead_result.sent:
        logger.error("Lead notification email failed for %s: %s", lead["request_id"], lead_result.error)
        services.update_email_status(lead["request_id"], email_error=f"notification: {lead_result.error}")
        warnings.append("internal notification email")

    confirmation_result = services.send_customer_confirmation_email(lead)
    services.update_email_status(lead["request_id"], confirmation_email_sent=confirmation_result.sent)
    if not confirmation_result.sent:
        logger.error("Confirmation email failed for %s: %s", lead["request_id"], confirmation_result.error)
        services.update_email_status(lead["request_id"], email_error=f"confirmation: {confirmation_result.error}")
        warnings.append("customer confirmation email")

    email_warning = None
    if warnings:
        email_warning = (
            "Your request was saved successfully, but the " + " and ".join(warnings) +
            " couldn't be sent right now. It's safely stored and can be retried."
        )

    return ServiceRequestResponse(
        request_id=lead["request_id"],
        status=lead["status"],
        created_at=lead["created_at"],
        lead_email_sent=lead_result.sent,
        confirmation_email_sent=confirmation_result.sent,
        email_warning=email_warning,
    )