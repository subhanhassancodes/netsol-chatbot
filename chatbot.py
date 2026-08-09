"""
chatbot.py

Holds the chatbot logic: RAG retrieval, Text2SQL, CSV analysis, and a
LangGraph agent that routes each question to the right source(s).

This module reads from the persistent stores built by ingest.py:
- ./chroma_db      (ChromaDB, website content)
- ./netsol_hr.db   (SQLite, employee data)

Architecture:
- Each node GATHERS raw context relevant to the question (website chunks,
  raw SQL rows, an uploaded document, or a computed pandas result from
  document_store.py) — it does NOT generate a user-facing answer itself.
- Exactly one final step, `_synthesize_answer()`, takes whatever context was
  gathered and produces ONE natural, conversational answer — no source
  labels ("Website info:" / "Employee data:"), no raw field/column dumps,
  and no mention of where the information came from unless the user
  explicitly asks. This is what lets a compound question (e.g. "who
  founded NETSOL and who's the highest-paid employee") come back as one
  smooth paragraph instead of two glued-together fragments.

Efficiency notes (Gemini's free tier has a low daily quota):
- router_node tries a fast, free keyword heuristic first, and only calls
  Gemini to classify when the question is genuinely ambiguous.
- A "NONE" route handles out-of-scope questions without wasting a call.
- Identical questions asked twice in the same run are served from an
  in-memory cache instead of calling Gemini again.
- Every raw Gemini call (routing, SQL/pandas generation, synthesis) is
  logged to token_tracker so usage and cost stay accurate no matter which
  path a question takes — see llm_client.py and token_tracker.py.

Import `app` and call app.invoke({...}), or use the convenience functions
`ask(question)` / `ask_detailed(question)` at the bottom.
"""

import re
import sqlite3
import time
import difflib
from typing import Optional, TypedDict

import chromadb
import numpy as np
import requests
import document_store
import services
import token_tracker
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from llm_client import client as _client_genai, MODEL_NAME, GEMINI_API_KEY, generate_with_retry as _generate_with_retry


# ---------- Setup ----------
CHROMA_DB_PATH = "./chroma_db"
SQLITE_DB_PATH = "netsol_hr.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PAKISTAN_DOMAIN = "www.netsolpk.com"

_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_collection = _chroma_client.get_collection("netsol_site")

_conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
_cursor = _conn.cursor()

_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GEMINI_API_KEY)


def _invoke_llm_tracked(prompt: str):
    """Wraps _llm.invoke() with the same token-usage logging that
    generate_with_retry() does for the raw genai client, so routing calls
    show up in the per-request and running totals too."""
    response = _llm.invoke(prompt)
    usage = response.usage_metadata or {}
    token_tracker.log_usage(
        MODEL_NAME,
        usage.get("input_tokens", 0) or 0,
        usage.get("output_tokens", 0) or 0,
    )
    return response


def _get_schema() -> str:
    _cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    return "\n\n".join(row[0] for row in _cursor.fetchall())


_SCHEMA = _get_schema()

# Simple in-memory cache so repeated identical questions don't burn quota.
_answer_cache: dict[str, dict] = {}

# session_id -> list of (question, answer) tuples, most recent last. Capped
# short — this exists purely to let the LLM resolve a follow-up question
# like "how much is his pay" (which has no meaning on its own), not to be a
# full transcript.
_session_history: dict[str, list[tuple[str, str]]] = {}
_MAX_HISTORY_TURNS = 4


def _history_text(session_id: Optional[str]) -> str:
    if not session_id:
        return ""
    turns = _session_history.get(session_id)
    if not turns:
        return ""
    lines = ["Recent conversation (use this ONLY to resolve pronouns/references like \"his\", \"that department\", \"those employees\" — the actual question to answer is given separately below):"]
    for q, a in turns:
        lines.append(f"Q: {q}")
        lines.append(f"A: {a}")
    return "\n".join(lines)


def _record_history(session_id: Optional[str], question: str, answer: str) -> None:
    if not session_id:
        return
    turns = _session_history.setdefault(session_id, [])
    turns.append((question, answer))
    del turns[:-_MAX_HISTORY_TURNS]


def reset_history(session_id: str) -> None:
    """Called when a session's uploaded document/CSV changes or is cleared,
    so stale conversation context doesn't bleed into questions about a
    different file."""
    _session_history.pop(session_id, None)
    _lead_sessions.pop(session_id, None)


# ---------- Conversational NETSOL services / lead-generation flow ----------
# The entire "interested in NETSOL's services" experience lives inside the
# normal chat — there is no separate form, page, or button-driven menu (see
# services.py for the catalog/storage/email pieces this flow uses). A
# per-session state machine tracks where in the conversation each session
# is; every reply is a short, natural chat message, exactly like any other
# chatbot turn.

_lead_sessions: dict[str, dict] = {}

# Cheap, deterministic pre-check for "this message expresses interest in
# NETSOL's services/solutions, or wants to talk to someone" — covers the
# natural variations called out in the spec, plus common everyday phrasings
# people actually type ("I want to do these services", "can I get services
# in AI"). Broad enough to catch real conversational intent, but still
# anchored on phrase patterns (not lone keywords like "service") so a plain
# factual question doesn't get hijacked.
_LEAD_TRIGGER_PATTERNS = [
    r"\binterested in (using )?(netsol'?s? )?(services?|solutions?)\b",
    r"\bwant to (use|do|get|try|have)\b.{0,40}\b(service|solution)s?\b",
    r"\bi want (these|this|that|the) (service|solution)s?\b",
    r"\bcan i (get|have|use)\b.{0,40}\b(service|solution)s?\b",
    r"\bi'?d like (these|this|that|the|to (use|get|try))\b.{0,40}\b(service|solution)s?\b",
    r"\bcan netsol help\b",
    r"\bwork with netsol\b",
    r"\blike to use\b.{0,40}\b(solution|service)s?\b",
    r"\bneed a netsol solution\b",
    r"\bnetsol solution for my (company|business)\b",
    r"\bspeak (to|with) (someone|a (netsol )?representative)\b",
    r"\btalk to a (netsol )?representative\b",
    r"\bwhat services does netsol (provide|offer)\b",
    r"\bsign (me )?up\b",
    r"\bget started\b.{0,40}\b(service|solution)s?\b",
]
_LEAD_TRIGGER_RE = re.compile("|".join(_LEAD_TRIGGER_PATTERNS), re.IGNORECASE)

# Smart/curly quotes (common from mobile keyboards and browser
# autocorrect — e.g. "NETSOL's" typed as "NETSOL’s") would otherwise silently
# fail every apostrophe-containing pattern above; normalize to a plain
# apostrophe before matching.
_SMART_QUOTES_RE = re.compile("[\u2018\u2019\u02bc]")


def _looks_like_lead_intent(question: str) -> bool:
    normalized = _SMART_QUOTES_RE.sub("'", question)
    return bool(_LEAD_TRIGGER_RE.search(normalized))


# Embed the real NETSOL catalog once so a user's free-text service
# description ("leasing software", "something for automotive finance") can
# be matched against it with the embedding model already loaded for RAG —
# no extra dependency, no LLM call needed just to match a service.
_SERVICE_CATALOG = services.list_services()
_SERVICE_TEXTS = [f"{s['name']}. {s['description']}" for s in _SERVICE_CATALOG]
_SERVICE_EMBEDDINGS = _embedding_model.encode(_SERVICE_TEXTS)


def _match_service(text: str, min_similarity: float = 0.34) -> Optional[dict]:
    query_emb = _embedding_model.encode([text])[0]
    denom = (np.linalg.norm(_SERVICE_EMBEDDINGS, axis=1) * np.linalg.norm(query_emb)) + 1e-9
    sims = (_SERVICE_EMBEDDINGS @ query_emb) / denom
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= min_similarity:
        return _SERVICE_CATALOG[best_idx]
    return None


def _service_menu_text() -> str:
    """A natural-language, bulleted rundown of the real NETSOL catalog,
    grouped by category — shown when someone expresses interest but hasn't
    named a specific service yet, so they have something concrete to react
    to instead of a blank "what do you want" prompt. Still plain chat text
    (the frontend renders "- " lines as a bullet list), not a card menu —
    the flow stays conversational either way."""
    by_category: dict[str, list[str]] = {cat: [] for cat in services.CATEGORIES}
    for s in _SERVICE_CATALOG:
        by_category.setdefault(s["category"], []).append(s["name"])

    blocks = []
    for category, names in by_category.items():
        if not names:
            continue
        bullet_lines = "\n".join(f"- {name}" for name in names)
        blocks.append(f"{category}\n{bullet_lines}")

    return "\n\n".join(blocks)


_YES_RE = re.compile(r"^\s*(yes|yeah|yup|sure|please|ok(ay)?|go ahead|sounds good|correct|that'?s right|absolutely|definitely|y)\b", re.IGNORECASE)
_NO_RE = re.compile(r"^\s*(no|nope|nah|not (now|really)|n)\b", re.IGNORECASE)
_SKIP_RE = re.compile(r"^\s*(skip|none|n/?a|not applicable|prefer not to say|rather not)\s*\.?\s*$", re.IGNORECASE)


def _parse_yes_no(text: str) -> Optional[bool]:
    t = text.strip()
    if _YES_RE.match(t):
        return True
    if _NO_RE.match(t):
        return False
    return None


def _is_skip(text: str) -> bool:
    return bool(_SKIP_RE.match(text.strip()))


def _start_lead_flow(session_id: str, question: str) -> str:
    matched = _match_service(question)
    state = {
        "stage": "service",
        "service_query": question,
        "matched_service": matched,
        "requirement": None,
        "name": None,
        "email": None,
        "company": None,
        "phone": None,
    }
    _lead_sessions[session_id] = state

    if matched:
        state["stage"] = "requirement"
        return (
            f"Absolutely — it sounds like you might be interested in {matched['name']}. "
            "Could you tell me a little about what your organization is looking for?"
        )
    return (
        "Absolutely, I'd be happy to help you explore NETSOL's services. Here's what we offer:\n\n"
        f"{_service_menu_text()}\n\n"
        "Which one are you interested in?"
    )


def _build_confirmation_summary(state: dict) -> str:
    service_label = state["matched_service"]["name"] if state["matched_service"] else state["service_query"]
    lead_in = f"Just to confirm — you're {state['name']}, reachable at {state['email']}"
    if state.get("company"):
        lead_in += f" from {state['company']}"
    return (
        f"{lead_in}, you're interested in {service_label}, and your requirement is: "
        f"\"{state['requirement']}\". A NETSOL representative will follow up with you. "
        "Does that all look correct?"
    )


def _finalize_lead(state: dict) -> str:
    service = state["matched_service"] or {"id": "general-inquiry", "name": state["service_query"]}
    lead = services.create_lead(
        name=state["name"],
        email=state["email"],
        service_id=service["id"],
        service_name=service["name"],
        description=state["requirement"],
        phone=state.get("phone"),
        company=state.get("company"),
    )

    lead_result = services.send_lead_notification_email(lead)
    services.update_email_status(lead["request_id"], lead_email_sent=lead_result.sent)
    if not lead_result.sent:
        services.update_email_status(lead["request_id"], email_error=f"notification: {lead_result.error}")

    confirmation_result = services.send_customer_confirmation_email(lead)
    services.update_email_status(lead["request_id"], confirmation_email_sent=confirmation_result.sent)
    if not confirmation_result.sent:
        services.update_email_status(lead["request_id"], email_error=f"confirmation: {confirmation_result.error}")

    reply = (
        "Thank you! Your request has been submitted successfully. A NETSOL representative "
        f"will follow up with you soon.\n\nRequest ID: {lead['request_id']}"
    )
    if not lead_result.sent or not confirmation_result.sent:
        reply += "\n\n(Your request is safely saved — a notification email had a hiccup, but the NETSOL team can still see it.)"
    return reply


def _continue_lead_flow(session_id: str, question: str) -> str:
    state = _lead_sessions[session_id]
    stage = state["stage"]
    text = question.strip()

    if stage == "service":
        matched = _match_service(text)
        state["matched_service"] = matched
        state["service_query"] = text
        state["stage"] = "requirement"
        if matched:
            return f"Great — {matched['name']} it is. Could you tell me a little about what your organization is looking for?"
        return "Got it. Could you tell me a little more about what your organization is looking for?"

    if stage == "requirement":
        state["requirement"] = text
        if state.get("editing"):
            # Correcting the service/requirement after a "no" at final
            # confirmation — contact details are already on file, so go
            # straight back to a fresh confirmation instead of re-asking
            # for name/email/etc. all over again.
            state["editing"] = False
            state["stage"] = "final_confirm"
            return _build_confirmation_summary(state)
        state["stage"] = "confirm_interest"
        return (
            "That sounds like something NETSOL may be able to help with. Would you like to "
            "submit your requirements to the NETSOL team so someone can follow up with you?"
        )

    if stage == "confirm_interest":
        if _parse_yes_no(text) is False:
            _lead_sessions.pop(session_id, None)
            return "No problem — happy to help with anything else in the meantime."
        state["stage"] = "name"
        return "Great, I'll just need a few details. What's your name?"

    if stage == "name":
        state["name"] = text
        state["stage"] = "email"
        first_name = text.split()[0] if text.split() else text
        return f"Thanks, {first_name}. What's the best email address to reach you?"

    if stage == "email":
        if not services.is_valid_email(text):
            return "That doesn't look like a valid email address — could you double-check it?"
        state["email"] = text
        state["stage"] = "company"
        return "What's your company or organization? (Feel free to say \"skip\" if you'd rather not share.)"

    if stage == "company":
        state["company"] = None if _is_skip(text) else text
        state["stage"] = "phone"
        return "And the best phone number to reach you, if you'd like to share one? (Or say \"skip\".)"

    if stage == "phone":
        state["phone"] = None if _is_skip(text) else text
        state["stage"] = "final_confirm"
        return _build_confirmation_summary(state)

    if stage == "final_confirm":
        if _parse_yes_no(text) is False:
            state["stage"] = "service"
            state["editing"] = True
            return "No worries — let's fix that. What NETSOL service are you interested in, and what's your requirement?"
        answer = _finalize_lead(state)
        _lead_sessions.pop(session_id, None)
        return answer

    # Safety net — should never be reached.
    _lead_sessions.pop(session_id, None)
    return "Let's start over — what NETSOL service are you interested in?"

# Load every stored chunk once, so we can also do a plain literal keyword
# search as a backstop to vector similarity. Vector search alone can miss a
# chunk that uses the exact word being asked about (e.g. "founder") if that
# chunk isn't semantically "close enough" to the phrasing of the question —
# a literal match guarantees it gets included regardless.
_all_docs_cache = _collection.get(include=["documents", "metadatas"])
_ALL_DOCS = [
    {"text": doc, "domain": meta.get("domain", "")}
    for doc, meta in zip(_all_docs_cache["documents"], _all_docs_cache["metadatas"])
]

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "who", "what", "when",
    "where", "why", "how", "does", "do", "did", "and", "or", "of", "for",
    "in", "on", "at", "to", "has", "have", "had", "this", "that", "it",
    "netsol", "netsols",
}


def _keyword_search(query: str, max_matches: int = 3, fuzzy_cutoff: float = 0.82) -> list[str]:
    """Returns chunks containing a match for any significant word in the
    question — exact match first, with a fuzzy fallback (via difflib) to
    catch small typos like 'fouder' instead of 'founder'. Cheap, local,
    no LLM/embedding call."""
    words = [w for w in re.findall(r"[a-z]+", query.lower()) if len(w) >= 4 and w not in _STOPWORDS]
    if not words:
        return []

    scored = []
    for doc in _ALL_DOCS:
        text_lower = doc["text"].lower()
        doc_words = set(re.findall(r"[a-z]+", text_lower))
        score = 0
        for w in words:
            if w in text_lower:
                score += 1
            elif difflib.get_close_matches(w, doc_words, n=1, cutoff=fuzzy_cutoff):
                score += 1
        if score > 0:
            scored.append((score, doc["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:max_matches]]


_GOOGLE_STATUS_URL = "https://status.cloud.google.com/incidents.json"
_RELEVANT_STATUS_KEYWORDS = ("generative", "gemini", "vertex ai", "ai platform")


def check_google_status() -> dict:
    """Checks Google's public Cloud status feed for active incidents related
    to Gemini/Generative Language/Vertex AI. Best-effort diagnostic helper."""
    try:
        resp = requests.get(_GOOGLE_STATUS_URL, timeout=5)
        resp.raise_for_status()
        incidents = resp.json()
    except Exception as e:
        return {"checked": False, "error": str(e), "active_incidents": []}

    active = []
    for incident in incidents:
        if incident.get("end") is not None:
            continue
        service_name = (incident.get("service_name") or "").lower()
        title = (incident.get("external_desc") or "").lower()
        if any(kw in service_name or kw in title for kw in _RELEVANT_STATUS_KEYWORDS):
            active.append({
                "service": incident.get("service_name"),
                "description": incident.get("external_desc"),
                "begin": incident.get("begin"),
                "url": incident.get("uri"),
            })

    return {"checked": True, "active_incidents": active}


# ---------- RAG side: gather relevant website chunks (no answer generation here) ----------
def _search(query: str, n_results: int = 3, domain_filter: Optional[str] = None):
    query_embedding = _embedding_model.encode([query])
    kwargs = {"query_embeddings": query_embedding.tolist(), "n_results": n_results}
    if domain_filter:
        kwargs["where"] = {"domain": domain_filter}
    return _collection.query(**kwargs)


def gather_rag_context(query: str, n_results: int = 8, max_total_chunks: int = 6) -> str:
    """Returns raw, relevant website text chunks joined together — NOT a
    generated answer. Combines a literal/fuzzy keyword backstop with
    Pakistan-priority vector search, capped at a small total so the final
    synthesis step isn't drowned in too much unrelated text."""
    keyword_chunks = _keyword_search(query, max_matches=3)

    remaining = max(0, max_total_chunks - len(keyword_chunks))
    pk_slots = (remaining + 1) // 2
    general_slots = remaining - pk_slots

    pk_results = _search(query, n_results=max(pk_slots, 1), domain_filter=PAKISTAN_DOMAIN)
    general_results = _search(query, n_results=max(general_slots, 1))

    pk_chunks = (pk_results["documents"][0] if pk_results["documents"] else [])[:pk_slots]
    general_chunks = (general_results["documents"][0] if general_results["documents"] else [])[:general_slots]

    seen = set()
    combined_chunks = []
    for chunk in keyword_chunks + pk_chunks + general_chunks:
        if chunk not in seen and len(combined_chunks) < max_total_chunks:
            combined_chunks.append(chunk)
            seen.add(chunk)

    return "\n\n".join(combined_chunks)


# ---------- Text2SQL side: gather raw rows (no answer generation here) ----------
def _question_to_sql(
    question: str,
    schema: str,
    previous_error: str = None,
    previous_sql: str = None,
    history_context: str = "",
) -> str:
    error_context = ""
    if previous_error:
        error_context = f"""

Your previous attempt failed. Fix it.
Previous SQL: {previous_sql}
Error: {previous_error}"""

    history_block = f"\n{history_context}\n" if history_context else ""

    prompt = f"""You are a SQL expert. Given the database schema below, write a SQLite query that answers the question.

Schema:
{schema}
{history_block}
Question: {question}

Rules:
- Return ONLY the SQL query, no explanation, no markdown formatting, no backticks.
- Use only the tables and columns shown in the schema.
- If joining tables, join on employee_id.
- For text fields that hold longer descriptive or free-text values (e.g. degree, job_title, address, summary, designation) — use LIKE '%value%' instead of exact equality, since these often contain extra words (e.g. "MS Data Science" contains "MS").
- For categorical/label-style text fields (e.g. nationality, marital_status, employee_status, current_department, current_team, revision_type, relationship) also use LIKE '%value%' rather than exact equality, and match on the SHORTEST distinctive root of the value the user gave rather than their exact wording — e.g. for "Pakistanis" use LIKE '%Pakistan%', not '%Pakistanis%' or '%Pakistani%', since the user's phrasing may be plural, singular, differently capitalized, or otherwise not byte-for-byte identical to the stored value (SQLite's LIKE is already case-insensitive for ASCII, but exact substrings must still actually appear).
- EXCEPTION: for the "gender" column, always use exact equality (gender = 'Male' or gender = 'Female'), NEVER LIKE. The word "Male" is a substring of "Female", so `gender LIKE '%Male%'` incorrectly matches female employees too — this has caused wrong gender counts before, so exact equality is mandatory here.
- EXCEPTION: for "employee_id", always use exact equality (employee_id = '0514'), NEVER LIKE. IDs are zero-padded strings (e.g. '0001', '0514') and a LIKE match on a partial or mistyped ID (e.g. '%7514%') can silently match a completely unrelated employee whose padded ID happens to contain that substring (e.g. '0514') — always compare the full, exact ID string the user gave, zero-padding it to match the column's format if needed, and if it plainly isn't a valid ID for this data, return a query that yields no rows rather than a fuzzy guess.
- If using SELECT DISTINCT, do NOT use ORDER BY on a column that isn't in the SELECT list — SQLite rejects that. Either include that column in the SELECT list too, or don't use DISTINCT.
- If the question contains multiple parts joined by "and" or similar, and only SOME of those parts relate to this employee database schema, write SQL that answers ONLY the relevant part. Completely ignore any part of the question that isn't about this schema (e.g. company history, founders, awards, website content) — do not try to represent it as a column, value, or WHERE condition.
- If the question uses a pronoun or implicit reference (e.g. "his pay", "that department", "those employees"), resolve WHO or WHAT it refers to using the recent conversation above before writing the query — do not guess or pick an arbitrary row.
- IMPORTANT: every row in the "employees" table is already a current NETSOL employee. To count or list all employees, just query the "employees" table directly (e.g. SELECT COUNT(*) FROM employees) — do NOT filter or join by a company name like "NETSOL" to do this. The "company" column in "job_experience" refers to a person's PAST employers (their work history before/outside NETSOL), which is a completely different thing — only use that table/column if the question is specifically asking about someone's previous employers.{error_context}

SQL query:"""

    sql = _generate_with_retry(prompt).strip()
    return sql.replace("```sql", "").replace("```", "").strip()


def _strip_distinct(sql: str) -> str:
    """Removes a leading DISTINCT after SELECT — the simplest fix for
    SQLite's 'ORDER BY term does not match any column' restriction."""
    return re.sub(r"(?i)\bSELECT\s+DISTINCT\b", "SELECT", sql, count=1)


def gather_sql_data(question: str, schema: str = _SCHEMA, history_context: str = "") -> dict:
    """Returns raw SQL results as data — NOT a formatted/generated answer —
    e.g. {"columns": [...], "rows": [...], "sql": "...", "error": None}.
    The final synthesis step turns this into natural language."""
    sql_query = _question_to_sql(question, schema, history_context=history_context)
    try:
        _cursor.execute(sql_query)
        rows = _cursor.fetchall()
        columns = [d[0] for d in _cursor.description]
        return {"columns": columns, "rows": rows, "sql": sql_query, "error": None}
    except Exception as first_error:
        error_text = str(first_error)

        # Known, deterministic fix: DISTINCT + ORDER BY on a non-selected
        # column. Just drop DISTINCT and retry — no LLM call needed.
        if "ORDER BY term does not match" in error_text:
            try:
                patched_sql = _strip_distinct(sql_query)
                _cursor.execute(patched_sql)
                rows = _cursor.fetchall()
                columns = [d[0] for d in _cursor.description]
                return {"columns": columns, "rows": rows, "sql": patched_sql, "error": None}
            except Exception as patch_error:
                error_text = f"{error_text} (also tried stripping DISTINCT: {patch_error})"

        # General fallback: give Gemini one chance to see the actual error
        # and fix its own query.
        try:
            fixed_sql = _question_to_sql(question, schema, previous_error=error_text, previous_sql=sql_query)
            _cursor.execute(fixed_sql)
            rows = _cursor.fetchall()
            columns = [d[0] for d in _cursor.description]
            return {"columns": columns, "rows": rows, "sql": fixed_sql, "error": None}
        except Exception as second_error:
            return {"columns": [], "rows": [], "sql": sql_query, "error": str(second_error)}


def gather_sql_data_multi(question: str, schema: str = _SCHEMA, history_context: str = "") -> list[dict]:
    """Splits a compound question into its SQL-relevant sub-questions (by
    "and") and runs a SEPARATE query for each one that independently
    matches an SQL keyword. This matters because a single SQL statement
    can't cleanly answer two different questions at once (e.g. "how many
    male employees are there and who is the highest paid employee" needs
    a COUNT query AND a MAX/ORDER-BY query) — asking one query to do both
    was silently answering only one part, or neither. Falls back to a
    single query against the whole question when it isn't actually
    compound, so simple questions don't pay for an extra LLM call."""
    segments = re.split(r"\band\b", question, flags=re.IGNORECASE)
    segments = [s.strip(" ,.;") for s in segments if s.strip(" ,.;")]

    sql_segments = [
        seg for seg in segments
        if _fuzzy_has_keyword(set(re.findall(r"[a-z]+", seg.lower())), _SQL_KEYWORDS)
    ]

    if len(sql_segments) <= 1:
        return [gather_sql_data(question, schema, history_context=history_context)]

    return [gather_sql_data(seg, schema, history_context=history_context) for seg in sql_segments]


# ---------- Final synthesis: the ONLY place a user-facing answer is written ----------
def _synthesize_answer(
    question: str,
    rag_context: Optional[str] = None,
    sql_data: Optional[list[dict]] = None,
    document_text: Optional[str] = None,
    csv_data: Optional[dict] = None,
    history_context: str = "",
) -> str:
    """Takes whatever raw context was gathered and produces ONE natural,
    conversational answer. This is the only function that generates
    user-facing text, which is what keeps compound answers from reading
    like two glued-together fragments with source labels.

    sql_data is a LIST of query results (see gather_sql_data_multi) since
    a compound HR question can require more than one SQL query — each
    result is included as its own block so the synthesis step has
    everything needed to answer every part of the question.

    history_context (see _history_text) carries the last few Q&A pairs
    for THIS session so a follow-up like "how much is his pay" can be
    phrased correctly even though the retrieved data itself has no notion
    of "his" — resolution of who/what a pronoun refers to already
    happened upstream (in SQL/CSV generation); this is a second pass so
    the final wording matches too."""

    sources_block = ""
    if history_context:
        sources_block += f"\n{history_context}\n"
    if document_text:
        sources_block += f"\n--- Uploaded document (treat as the primary source of truth) ---\n{document_text}\n"
    if csv_data is not None:
        if csv_data.get("error"):
            sources_block += (
                f"\n--- Uploaded spreadsheet ---\n(A calculation was attempted but failed: {csv_data['error']})\n"
            )
        else:
            sources_block += (
                "\n--- Uploaded spreadsheet (treat as the primary source of truth; this is the exact, "
                "already-computed result of a real calculation against the full dataset — report it "
                "precisely, don't recompute, round differently, or second-guess it) ---\n"
                f"{csv_data['result']}\n"
            )
    if rag_context:
        sources_block += f"\n--- Company website content ---\n{rag_context}\n"
    if sql_data:
        multi = len(sql_data) > 1
        for i, data in enumerate(sql_data, start=1):
            label = f"--- Employee database (part {i} of {len(sql_data)}) ---" if multi else "--- Employee database ---"
            if data.get("error"):
                sources_block += f"\n{label}\n(A query was attempted but failed: {data['error']})\n"
            elif not data["rows"]:
                sources_block += f"\n{label}\nQuery run: {data['sql']}\n(No matching records were found for this part of the question.)\n"
            else:
                rows_text = "\n".join(
                    ", ".join(f"{col}={val}" for col, val in zip(data["columns"], row))
                    for row in data["rows"]
                )
                # The query text itself is included (not just the raw result) so the
                # model knows exactly what was counted/filtered — e.g. a bare
                # "COUNT(*)=361" is ambiguous on its own (is 361 a total, or a
                # count that was already filtered to gender='Male'?), and that
                # ambiguity was causing wrong or hedging answers for filtered
                # aggregate questions.
                sources_block += f"\n{label}\nQuery run: {data['sql']}\nColumns: {', '.join(data['columns'])}\n{rows_text}\n"

    if not sources_block.strip():
        sources_block = "(No relevant information was found from any source.)"

    prompt = f"""You are a helpful assistant answering questions about NETSOL Technologies. Use the information below to answer the user's question.

{sources_block}

Question: {question}

How to answer:
- Write ONE natural, conversational, well-written answer — the way a knowledgeable person would answer directly, not a report.
- A "Query run: ..." line above a result tells you exactly what was counted/filtered/matched (e.g. it distinguishes a total count from a count already filtered to one gender or department) — use it to interpret the number correctly, but NEVER quote, paraphrase, or refer to the query itself in your answer to the user.
- Lead with the direct answer in the first sentence, then add only the supporting detail that's actually relevant — no filler, no repeating the question back, no padding a short fact into extra sentences. For a general/factual question, aim for a few tight sentences rather than a long paragraph. If the user is specifically asking to see a list of records or a detailed breakdown, give the full detail they asked for — conciseness means cutting unnecessary words, not cutting requested information.
- NEVER mention where the information came from (don't say "website", "database", "SQL", "CSV", "spreadsheet", "document", "context", "data file", "records", etc.) unless the user's question explicitly asks about the source itself.
- NEVER use section labels or headers like "Website info:", "Employee data:", "Answer:", etc. Just answer.
- NEVER show raw field/column names, "Column: value" style output, or raw code/table dumps (e.g. don't say "Employee Count: 514" or "employee_name: Danish Thomas") — turn it into a fluent sentence (e.g. "NETSOL currently has 514 employees." or "Danish Thomas is the highest-paid employee."). Exception: if the uploaded spreadsheet result is itself a list of multiple records the user asked to see, a short clean list is fine.
- The context may describe things indirectly — for example, a bio that says someone "founded" or was involved in "the founding of" a company means that person is a founder, and a page titled "AI & ML Services" is relevant to a question about an "AI department". Make reasonable connections like this.
- If the question has multiple parts, answer ALL of the parts you have information for, woven into one smooth response — don't ignore a part just because the information for it came from a different source than the other part.
- If you genuinely have no relevant information for part (or all) of the question, say so briefly and naturally, without dwelling on it or explaining why.
- If the question uses a pronoun or implicit reference (e.g. "his pay", "that department"), the recent conversation above (if present) tells you who/what it refers to — use it to phrase the answer correctly rather than guessing.

Answer:"""

    return _generate_with_retry(prompt)


# ---------- Keyword pre-routing (free, no LLM call) ----------
_SQL_KEYWORDS = {
    "employee", "employees", "salary", "salaries", "department", "designation",
    "education", "degree", "skill", "skills", "job", "experience", "hired",
    "headcount", "team", "grade", "promotion", "promotions", "cnic", "gender",
    "joined", "joining", "staff", "worker", "workers", "payroll", "institution",
    "nationality", "nationalities", "certification", "certifications",
    "language", "languages", "tool", "tools",
    "training", "referee", "referees", "marital", "email", "mobile", "phone",
    "address", "father", "probation", "termination",
}

_RAG_KEYWORDS = {
    "award", "awards", "history", "mission", "investor", "investors", "board",
    "director", "directors", "news", "contact", "governance", "netsol",
    "company", "about", "vision", "website", "csr", "media", "institute",
    "founder", "founders", "founded", "ceo", "chairman", "chairperson",
    "chief", "executive", "leadership", "headquarters", "established",
    "incorporated", "listed", "subsidiary", "subsidiaries", "office",
    "offices", "location", "locations", "partnership", "partnerships",
    "client", "clients", "product", "products", "solution", "solutions",
    "technology", "service", "services", "global", "presence",
}

# Words too generic to reliably signal "this is a website question" on their
# own for a single-topic question (they appear in almost every sentence
# about the company). Only trusted as a RAG signal in genuinely compound
# questions (multiple clauses joined by "and").
_GENERIC_RAG_WORDS = {"netsol", "company", "about", "website"}


# "department"/"team" are genuinely ambiguous on their own: they signal an
# SQL question when asking about an employee's ASSIGNMENT ("who's on the
# WEOM team", "how many people are in Finance") but a RAG/website question
# when asking what a department/team DOES or is responsible for ("what does
# the AI department do") — a bag-of-words match can't tell those apart, and
# treating a lone match on one of these as a confident SQL signal was
# exactly why department-function questions were being misrouted to the
# employee database instead of the website content. So a lone match on
# either word is deliberately NOT treated as a strong SQL signal here —
# it falls through to the LLM classifier below instead, which reads the
# whole question and can actually distinguish "what does X do" (RAG) from
# "who's in X" (SQL). See router_node()'s prompt for that distinction.
_AMBIGUOUS_SQL_WORDS = {"department", "departments", "team", "teams"}


def _matched_keywords(words: set, keyword_set: set, cutoff: float = 0.8) -> set:
    matched = set()
    for w in words:
        if w in keyword_set:
            matched.add(w)
            continue
        close = difflib.get_close_matches(w, keyword_set, n=1, cutoff=cutoff)
        if close:
            matched.add(close[0])
    return matched


def _fuzzy_has_keyword(words: set, keyword_set: set, cutoff: float = 0.8) -> bool:
    return bool(_matched_keywords(words, keyword_set, cutoff))


def _strong_sql_hit(words: set) -> bool:
    """True only if a match exists on an SQL keyword OTHER than the
    ambiguous department/team words — see _AMBIGUOUS_SQL_WORDS above."""
    return bool(_matched_keywords(words, _SQL_KEYWORDS) - _AMBIGUOUS_SQL_WORDS)


def _keyword_route(question: str) -> Optional[str]:
    """Returns 'SQL', 'RAG', 'BOTH', or None if the question doesn't clearly match one."""
    segments = re.split(r"\band\b", question, flags=re.IGNORECASE)
    segments = [s.strip(" ,.;") for s in segments if s.strip(" ,.;")]

    # The generic words below (netsol/company/about/website) are subtracted
    # from every RAG check, not just single-segment questions — otherwise a
    # compound SQL question that happens to mention "NETSOL" (e.g. "how many
    # female and male employees are at netsol") gets falsely tagged as also
    # being a website question, which then causes _split_compound to trim
    # words out of the SQL side for no reason.
    words_whole = set(re.findall(r"[a-z]+", question.lower()))
    sql_hit = _strong_sql_hit(words_whole)
    rag_hit = _fuzzy_has_keyword(words_whole, _RAG_KEYWORDS - _GENERIC_RAG_WORDS)

    if len(segments) > 1:
        for seg in segments:
            seg_words = set(re.findall(r"[a-z]+", seg.lower()))
            if _strong_sql_hit(seg_words):
                sql_hit = True
            if _fuzzy_has_keyword(seg_words, _RAG_KEYWORDS - _GENERIC_RAG_WORDS):
                rag_hit = True

    if sql_hit and rag_hit:
        return "BOTH"
    if sql_hit:
        return "SQL"
    if rag_hit:
        return "RAG"
    return None


# ---------- LangGraph state + nodes ----------
class ChatState(TypedDict):
    question: str
    route: Optional[str]
    answer: Optional[str]
    history_context: str


NONE_ANSWER = (
    "I can only answer questions about NETSOL's website (history, awards, mission, "
    "investors, etc.) or NETSOL's employee data (salaries, departments, skills, etc.). "
    "Try asking something along those lines!"
)


def router_node(state: ChatState) -> ChatState:
    route = _keyword_route(state["question"])
    if route is not None:
        return {"route": route}

    prompt = f"""Classify this question into exactly one category:
- "SQL" if it's ONLY about employee-level data: salaries, departments/teams an employee is assigned to, headcount, education, job history, skills, designations — anything that requires looking up or counting individual employee records
- "RAG" if it's ONLY about the company/website itself: NETSOL's history, awards, mission, investors, board of directors, news, or what a department/team/division DOES or is responsible for (its role, function, purpose — not who's in it)
- "BOTH" if it asks about employee data AND website/company information together (e.g. two questions joined by "and")
- "NONE" if it's unrelated to either (e.g. small talk, general knowledge, greetings)

A question like "what does the AI department do" is RAG (asking about the department's function), while "how many people are in the AI department" or "who is on the AI team" is SQL (asking about employee assignment/headcount).

Question: {state['question']}

Answer with just one word: SQL, RAG, BOTH, or NONE"""

    last_error = None
    for attempt in range(4):
        try:
            response = _invoke_llm_tracked(prompt)
            result = response.content.strip().upper()
            return {"route": result if result in ("SQL", "RAG", "BOTH", "NONE") else "NONE"}
        except Exception as e:
            last_error = e
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                time.sleep(2.0 * (2 ** attempt))
                continue
            raise
    raise last_error


def rag_node(state: ChatState) -> ChatState:
    context = gather_rag_context(state["question"])
    return {"answer": _synthesize_answer(state["question"], rag_context=context, history_context=state.get("history_context", ""))}


def sql_node(state: ChatState) -> ChatState:
    history_context = state.get("history_context", "")
    data = gather_sql_data_multi(state["question"], _SCHEMA, history_context=history_context)
    return {"answer": _synthesize_answer(state["question"], sql_data=data, history_context=history_context)}


def both_node(state: ChatState) -> ChatState:
    history_context = state.get("history_context", "")
    rag_context = gather_rag_context(state["question"])  # RAG sees the full question
    sql_data = gather_sql_data_multi(state["question"], _SCHEMA, history_context=history_context)  # split internally
    answer = _synthesize_answer(state["question"], rag_context=rag_context, sql_data=sql_data, history_context=history_context)
    return {"answer": answer}


def none_node(state: ChatState) -> ChatState:
    return {"answer": NONE_ANSWER}


def decide_route(state: ChatState) -> str:
    if state["route"] == "SQL":
        return "sql_node"
    elif state["route"] == "RAG":
        return "rag_node"
    elif state["route"] == "BOTH":
        return "both_node"
    return "none_node"


_graph = StateGraph(ChatState)
_graph.add_node("router_node", router_node)
_graph.add_node("rag_node", rag_node)
_graph.add_node("sql_node", sql_node)
_graph.add_node("both_node", both_node)
_graph.add_node("none_node", none_node)
_graph.set_entry_point("router_node")
_graph.add_conditional_edges(
    "router_node",
    decide_route,
    {"rag_node": "rag_node", "sql_node": "sql_node", "both_node": "both_node", "none_node": "none_node"},
)
_graph.add_edge("rag_node", END)
_graph.add_edge("sql_node", END)
_graph.add_edge("both_node", END)
_graph.add_edge("none_node", END)

app = _graph.compile()


# ---------- Uploaded document / CSV Q&A ----------
def ask_detailed(question: str, session_id: Optional[str] = None) -> dict:
    """Returns {"route": ..., "answer": ..., "cached": bool}.

    If session_id has an uploaded CSV, it's queried with real pandas
    computation (see document_store.py) and used as the primary context.
    Else if session_id has an uploaded text document, that text is used as
    the primary context. Either way normal RAG/SQL routing is skipped so
    the uploaded data stays the source of truth for every follow-up
    question until a new file is uploaded or the document is cleared. With
    no session_id or no uploaded document, behavior is unchanged.

    Every path also pulls in this session's recent conversation (see
    _history_text) so a follow-up question that only makes sense in
    context — "how much is his pay", "what about that department" — gets
    resolved against who/what was actually just discussed, instead of the
    model guessing at an arbitrary row."""
    history_context = _history_text(session_id)

    # Highest priority: a service-request conversation already in progress
    # for this session picks up exactly where it left off, regardless of
    # what the message would otherwise route to.
    if session_id and session_id in _lead_sessions:
        answer = _continue_lead_flow(session_id, question)
        _record_history(session_id, question, answer)
        return {"route": "LEAD", "answer": answer, "cached": False}

    if session_id and document_store.has_csv(session_id):
        csv_data = document_store.gather_csv_answer(session_id, question, history_context=history_context)
        answer = _synthesize_answer(question, csv_data=csv_data, history_context=history_context)
        _record_history(session_id, question, answer)
        return {"route": "CSV", "answer": answer, "cached": False}

    if session_id and document_store.has_document(session_id):
        document_text = document_store.get_document_text(session_id)
        if document_text is not None:
            answer = _synthesize_answer(question, document_text=document_text, history_context=history_context)
            _record_history(session_id, question, answer)
            return {"route": "DOCUMENT", "answer": answer, "cached": False}

    # A fresh expression of commercial/service interest starts the
    # conversational lead flow instead of going through normal RAG/SQL
    # routing — see the module docstring section above. Needs a session_id
    # to track multi-turn state.
    if session_id and _looks_like_lead_intent(question):
        answer = _start_lead_flow(session_id, question)
        _record_history(session_id, question, answer)
        return {"route": "LEAD", "answer": answer, "cached": False}

    # The answer cache is keyed by question text only, across all sessions —
    # fine for a fixed factual question, but wrong for anything that could
    # depend on THIS session's conversation so far, so skip it whenever
    # there's active history to consider.
    normalized = question.strip().lower()
    if not history_context and normalized in _answer_cache:
        cached = dict(_answer_cache[normalized])
        cached["cached"] = True
        return cached

    result = app.invoke({"question": question, "route": None, "answer": None, "history_context": history_context})
    output = {"route": result["route"], "answer": result["answer"], "cached": False}
    if not history_context:
        _answer_cache[normalized] = {"route": output["route"], "answer": output["answer"]}
    _record_history(session_id, question, output["answer"])
    return output


def ask(question: str, session_id: Optional[str] = None) -> str:
    """Convenience wrapper: ask a question, get just the answer string."""
    return ask_detailed(question, session_id=session_id)["answer"]


if __name__ == "__main__":
    print(ask_detailed("How many employees are in the Transcend department?"))
    print("---")
    print(ask_detailed("What awards has NETSOL won?"))
    print("---")
    print(ask_detailed("Who is the founder of NETSOL and who is the highest paid employee?"))
    print("---")
    print(ask_detailed("What's the weather like today?"))

    