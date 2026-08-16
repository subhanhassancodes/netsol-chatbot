"""
services.py

Everything the NETSOL services discovery / lead-generation feature needs,
in one module:
  1. SERVICES catalog        - the real NETSOL offerings shown as cards
  2. Lead storage (SQLite)   - netsol_leads.db
  3. Email sending           - internal notification + customer confirmation

Kept together (rather than split into a catalog/store/email file each)
since this is the entire feature's backend and there's no other module in
the project that needs any one piece of it independently. main.py is the
only caller.

======================================================================
1. SERVICES CATALOG
======================================================================
Every name and description below is taken directly from NETSOL's own
official pages (netsoltech.com) — specifically the exact pages already
scraped into the RAG store by ingest.py's EXTRA_URLS list (see
netsol_scraped.json). Nothing here is invented; descriptions are lightly
condensed from each page's own hero copy for card-length display. If
NETSOL's site adds/renames/removes an offering, update the entries below
(and the "url" field) to match.
"""

CATEGORIES = ["Finance & Leasing Platforms", "Technology Services", "Industry Solutions"]

SERVICES = [
    # ---- Products (Finance & Leasing Platforms) ----
    {
        "id": "transcend",
        "name": "Transcend Platform",
        "description": "AI-enabled asset finance platform for captives and lenders, from origination to digital retail.",
        "category": "Finance & Leasing Platforms",
        "url": "https://netsoltech.com/products/transcend",
    },
    {
        "id": "digital-retail",
        "name": "Transcend Retail (Digital Retail)",
        "description": "Automotive digital retail for OEMs and dealers, driving omnichannel, AI-enabled buying experiences.",
        "category": "Finance & Leasing Platforms",
        "url": "https://netsoltech.com/products/digital-retail",
    },
    {
        "id": "portals",
        "name": "Intermediary Portals",
        "description": "Bespoke origination portals for brokers, lenders, and dealers with a unified purchasing experience.",
        "category": "Finance & Leasing Platforms",
        "url": "https://netsoltech.com/products/portals",
    },
    {
        "id": "loan-origination",
        "name": "Loan Origination",
        "description": "Unified, AI-enabled origination platform covering the full quote-to-funding journey.",
        "category": "Finance & Leasing Platforms",
        "url": "https://netsoltech.com/products/loan-origination",
    },
    {
        "id": "lease-servicing",
        "name": "Lease & Contract Servicing",
        "description": "Manage loan and lease portfolios from activation through end-of-term with efficiency and visibility.",
        "category": "Finance & Leasing Platforms",
        "url": "https://netsoltech.com/products/lease-servicing",
    },
    {
        "id": "wholesale-finance",
        "name": "Wholesale Finance",
        "description": "Automate wholesale finance and floor-planning operations with AI-enabled credit and contract tools.",
        "category": "Finance & Leasing Platforms",
        "url": "https://netsoltech.com/products/wholesale-finance",
    },
    {
        "id": "mobility-solutions",
        "name": "Mobility Solutions",
        "description": "Microleasing and fractional ownership with an omni-channel, user-centric retail experience.",
        "category": "Finance & Leasing Platforms",
        "url": "https://netsoltech.com/products/mobility-solutions",
    },
    # ---- Services (Technology Services) ----
    {
        "id": "information-security",
        "name": "Information Security",
        "description": "SIEM, SOAR, and managed SOC services securing your digital environment with AI-enabled threat detection.",
        "category": "Technology Services",
        "url": "https://netsoltech.com/services/information-security",
    },
    {
        "id": "digital-solutions",
        "name": "Digital Solutions",
        "description": "Digital transformation and resource augmentation services backed by strategic talent partnerships.",
        "category": "Technology Services",
        "url": "https://netsoltech.com/services/digital-solutions",
    },
    {
        "id": "ai-ml-services",
        "name": "AI & ML Services",
        "description": "Turn your data into insights with data analytics and predictive intelligence powered by AI/ML.",
        "category": "Technology Services",
        "url": "https://netsoltech.com/services/ai-ml-services-and-solutions",
    },
    {
        "id": "generative-ai",
        "name": "Generative AI Services",
        "description": "AI consulting and generative AI solutions to transform your business landscape.",
        "category": "Technology Services",
        "url": "https://netsoltech.com/services/generative-ai",
    },
    {
        "id": "emerging-technologies",
        "name": "Emerging Technologies (Web 3.0)",
        "description": "Smart contracts, dApps, digital twins, and other Web 3.0 services for enterprises.",
        "category": "Technology Services",
        "url": "https://netsoltech.com/services/emerging-technologies",
    },
    {
        "id": "cloud-services",
        "name": "Cloud Services",
        "description": "Multi-cloud networking, managed infrastructure, and DevOps to scale efficiently and reduce costs.",
        "category": "Technology Services",
        "url": "https://netsoltech.com/services/cloud-services",
    },
    {
        "id": "data-engineering",
        "name": "Data Engineering",
        "description": "Scalable data pipelines, ETL, and data warehousing services for business growth.",
        "category": "Technology Services",
        "url": "https://netsoltech.com/services/data-engineering",
    },
    # ---- Solutions (Industry Solutions) ----
    {
        "id": "asset-finance",
        "name": "Asset Finance Solutions",
        "description": "Unified, AI-enabled asset finance platform serving lenders and leasing companies for 40+ years.",
        "category": "Industry Solutions",
        "url": "https://netsoltech.com/solutions/asset-finance",
    },
    {
        "id": "automotive-finance",
        "name": "Automotive Finance Solutions",
        "description": "AI-enabled loan origination, contract management, and digital retail tailored for automotive lenders and dealers.",
        "category": "Industry Solutions",
        "url": "https://netsoltech.com/solutions/automotive-finance",
    },
    {
        "id": "equipment-finance",
        "name": "Equipment Finance Solutions",
        "description": "End-to-end equipment finance technology for OEMs, lenders, and lessors across asset types.",
        "category": "Industry Solutions",
        "url": "https://netsoltech.com/solutions/equipment-finance",
    },
]

# Not part of the browsable catalog (list_services() below never returns
# it) — used only when someone picks "I'd like to speak with a NETSOL
# representative" in the entry-point step, which skips service browsing
# entirely and goes straight to the requirement form.
_GENERAL_INQUIRY = {
    "id": "general-inquiry",
    "name": "General Inquiry / Speak with a Representative",
    "description": "Not tied to a specific service — a NETSOL representative will follow up directly.",
    "category": "",
    "url": "https://www.netsolpk.com/contact-us",
}

_BY_ID = {s["id"]: s for s in SERVICES}
_BY_ID[_GENERAL_INQUIRY["id"]] = _GENERAL_INQUIRY


def list_services() -> list[dict]:
    """Returns the full browsable catalog (excludes the general-inquiry entry)."""
    return SERVICES


def get_service(service_id: str) -> dict | None:
    return _BY_ID.get(service_id)


"""
======================================================================
2. LEAD STORAGE (SQLite)
======================================================================
Leads live in their own database file — netsol_leads.db — separate from
netsol_hr.db. Why: ingest.py rebuilds netsol_hr.db from the employee JSON
export every time it's re-run, and — by design (see build_sql_store() in
ingest.py) — drops every table in that file except "employees" first, so
stale tables never linger. A "service_requests" table living in that same
file would be silently destroyed the next time someone re-runs ingestion
to refresh employee/website data. Leads are business data, not
derived/rebuildable data, so they get their own persistent file that
nothing else ever drops or rewrites.

Schema (service_requests):
  request_id      TEXT PRIMARY KEY   e.g. "NETSOL-2026-000123"
  name, email, phone, company        customer-supplied fields
  service_id      TEXT NOT NULL      slug from SERVICES above
  service_name    TEXT NOT NULL      display name, captured at submit time
                                     so a later catalog edit doesn't change history
  description     TEXT NOT NULL      the customer's stated requirement
  status          TEXT NOT NULL      "New" initially; free-form after that,
                                     so this is easy to extend into a full
                                     admin/lead-management dashboard later
  created_at      TEXT NOT NULL      ISO 8601 UTC
  lead_email_sent, confirmation_email_sent   INTEGER 0/1
  email_error     TEXT               last email error, if any, for retry/debugging
"""

import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Resolved relative to this file, not the current working directory, so it
# doesn't matter whether the app is started from the project root or from
# inside backend/ — netsol_leads.db lives at the project root, one level
# up from this file (backend/services.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEADS_DB_PATH = str(_PROJECT_ROOT / "netsol_leads.db")

_lock = threading.Lock()
_conn = sqlite3.connect(LEADS_DB_PATH, check_same_thread=False)


def _init_leads_db() -> None:
    with _lock:
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS service_requests (
                request_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT,
                company TEXT,
                service_id TEXT NOT NULL,
                service_name TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'New',
                created_at TEXT NOT NULL,
                lead_email_sent INTEGER NOT NULL DEFAULT 0,
                confirmation_email_sent INTEGER NOT NULL DEFAULT 0,
                email_error TEXT
            )
            """
        )
        _conn.commit()


_init_leads_db()


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    return bool(email) and bool(_EMAIL_RE.match(email.strip()))


def _next_request_id() -> str:
    """Generates a request ID like NETSOL-2026-000123 — year of creation,
    then a zero-padded sequence number that resets each year."""
    year = datetime.now(timezone.utc).year
    with _lock:
        cur = _conn.execute(
            "SELECT COUNT(*) FROM service_requests WHERE request_id LIKE ?",
            (f"NETSOL-{year}-%",),
        )
        count = cur.fetchone()[0]
    return f"NETSOL-{year}-{count + 1:06d}"


def create_lead(
    name: str,
    email: str,
    service_id: str,
    service_name: str,
    description: str,
    phone: Optional[str] = None,
    company: Optional[str] = None,
) -> dict:
    """Inserts a new lead with status "New" and returns the full stored
    row as a dict (including the generated request_id and created_at)."""
    request_id = _next_request_id()
    created_at = datetime.now(timezone.utc).isoformat()

    row = {
        "request_id": request_id,
        "name": name.strip(),
        "email": email.strip(),
        "phone": (phone or "").strip() or None,
        "company": (company or "").strip() or None,
        "service_id": service_id,
        "service_name": service_name,
        "description": description.strip(),
        "status": "New",
        "created_at": created_at,
        "lead_email_sent": 0,
        "confirmation_email_sent": 0,
        "email_error": None,
    }

    with _lock:
        _conn.execute(
            """
            INSERT INTO service_requests
                (request_id, name, email, phone, company, service_id,
                 service_name, description, status, created_at,
                 lead_email_sent, confirmation_email_sent, email_error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["request_id"], row["name"], row["email"], row["phone"],
                row["company"], row["service_id"], row["service_name"],
                row["description"], row["status"], row["created_at"],
                row["lead_email_sent"], row["confirmation_email_sent"], row["email_error"],
            ),
        )
        _conn.commit()

    return row


def update_email_status(
    request_id: str,
    lead_email_sent: Optional[bool] = None,
    confirmation_email_sent: Optional[bool] = None,
    email_error: Optional[str] = None,
) -> None:
    """Called after attempting to send either email — the lead row itself
    is never touched/deleted here, only these status fields, so a failed
    email never loses the underlying lead."""
    sets, params = [], []
    if lead_email_sent is not None:
        sets.append("lead_email_sent = ?")
        params.append(1 if lead_email_sent else 0)
    if confirmation_email_sent is not None:
        sets.append("confirmation_email_sent = ?")
        params.append(1 if confirmation_email_sent else 0)
    if email_error is not None:
        sets.append("email_error = ?")
        params.append(email_error)
    if not sets:
        return
    params.append(request_id)

    with _lock:
        _conn.execute(f"UPDATE service_requests SET {', '.join(sets)} WHERE request_id = ?", params)
        _conn.commit()


def get_lead(request_id: str) -> Optional[dict]:
    with _lock:
        cur = _conn.execute("SELECT * FROM service_requests WHERE request_id = ?", (request_id,))
        row = cur.fetchone()
        columns = [d[0] for d in cur.description]
    return dict(zip(columns, row)) if row else None


def list_leads(limit: int = 100) -> list[dict]:
    """Most recent leads first — the foundation for a future admin/lead
    dashboard; not exposed via any API route yet."""
    with _lock:
        cur = _conn.execute(
            "SELECT * FROM service_requests ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description]
    return [dict(zip(columns, r)) for r in rows]


"""
======================================================================
3. EMAIL SENDING
======================================================================
Sends the two emails triggered by a submitted service request:
  1. An internal lead-notification email to NETSOL's configured recipient.
  2. A confirmation email to the customer who submitted the request.

All SMTP configuration comes from environment variables — nothing is
hard-coded. Add these to your .env:

  SMTP_HOST               e.g. smtp.gmail.com
  SMTP_PORT               e.g. 587
  SMTP_USERNAME           the account used to send mail
  SMTP_PASSWORD           an app password (never your real account password)
  SMTP_FROM_EMAIL         optional — defaults to SMTP_USERNAME
  SMTP_FROM_NAME          optional — defaults to "NETSOL Assistant"
  LEAD_NOTIFICATION_EMAIL optional — defaults to subhanhassan999@gmail.com

If SMTP isn't configured at all (no SMTP_HOST), both send functions return
a clear "not configured" result instead of raising, so the rest of the
request flow (storing the lead) still works during local development
without email set up.
"""

import os
import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

# _PROJECT_ROOT is already defined above (near LEADS_DB_PATH) — reused here
# so .env resolves the same explicit way regardless of where this module's
# code runs from.
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL", SMTP_USERNAME)
SMTP_FROM_NAME = os.environ.get("SMTP_FROM_NAME", "NETSOL Assistant")

# Where every new lead notification goes. Configurable, but defaults to
# the address requested for this feature.
LEAD_NOTIFICATION_EMAIL = os.environ.get("LEAD_NOTIFICATION_EMAIL", "subhanhassan999@gmail.com")


@dataclass
class EmailResult:
    sent: bool
    error: Optional[str] = None


def _smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD)


def _send_email(to_email: str, subject: str, html_body: str, text_body: str) -> EmailResult:
    if not _smtp_configured():
        return EmailResult(sent=False, error="SMTP is not configured (missing SMTP_HOST/SMTP_USERNAME/SMTP_PASSWORD).")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
    msg["To"] = to_email
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_EMAIL, [to_email], msg.as_string())
        return EmailResult(sent=True)
    except Exception as e:
        return EmailResult(sent=False, error=str(e))


def send_lead_notification_email(lead: dict) -> EmailResult:
    """Notifies NETSOL (LEAD_NOTIFICATION_EMAIL) about a new service
    request. Never includes the customer's own inbox — this is the
    internal-facing email."""
    subject = f"New NETSOL Service Request — {lead['request_id']}"

    text_body = f"""New NETSOL service request received.

Request ID: {lead['request_id']}
Status: {lead['status']}
Submitted: {lead['created_at']}

Customer Name: {lead['name']}
Customer Email: {lead['email']}
Phone: {lead.get('phone') or '(not provided)'}
Company: {lead.get('company') or '(not provided)'}

Service Requested: {lead['service_name']}

Requirement:
{lead['description']}
"""

    html_body = f"""
<div style="font-family: -apple-system, Arial, sans-serif; max-width: 560px; margin: 0 auto;">
  <div style="background: linear-gradient(135deg, #1b75bb, #124d78); padding: 20px 24px; border-radius: 10px 10px 0 0;">
    <h2 style="color: #fff; margin: 0; font-size: 18px;">New NETSOL Service Request</h2>
    <p style="color: #cfe3f3; margin: 4px 0 0 0; font-size: 13px;">{lead['request_id']}</p>
  </div>
  <div style="border: 1px solid #e2e2e2; border-top: none; border-radius: 0 0 10px 10px; padding: 20px 24px;">
    <table style="width: 100%; border-collapse: collapse; font-size: 14px; color: #222;">
      <tr><td style="padding: 6px 0; color: #667; width: 140px;">Name</td><td style="padding: 6px 0;">{lead['name']}</td></tr>
      <tr><td style="padding: 6px 0; color: #667;">Email</td><td style="padding: 6px 0;">{lead['email']}</td></tr>
      <tr><td style="padding: 6px 0; color: #667;">Phone</td><td style="padding: 6px 0;">{lead.get('phone') or '&mdash;'}</td></tr>
      <tr><td style="padding: 6px 0; color: #667;">Company</td><td style="padding: 6px 0;">{lead.get('company') or '&mdash;'}</td></tr>
      <tr><td style="padding: 6px 0; color: #667;">Service</td><td style="padding: 6px 0;"><b>{lead['service_name']}</b></td></tr>
      <tr><td style="padding: 6px 0; color: #667;">Submitted</td><td style="padding: 6px 0;">{lead['created_at']}</td></tr>
      <tr><td style="padding: 6px 0; color: #667;">Status</td><td style="padding: 6px 0;">{lead['status']}</td></tr>
    </table>
    <p style="color: #667; margin: 16px 0 4px 0; font-size: 13px;">Requirement</p>
    <p style="background: #f5f7fa; border-radius: 8px; padding: 12px 14px; font-size: 14px; color: #222; white-space: pre-wrap;">{lead['description']}</p>
  </div>
</div>
"""

    return _send_email(LEAD_NOTIFICATION_EMAIL, subject, html_body, text_body)


def send_customer_confirmation_email(lead: dict) -> EmailResult:
    """Confirms receipt with the customer. Does NOT expose the internal
    LEAD_NOTIFICATION_EMAIL address anywhere in this email."""
    subject = f"We've received your request — {lead['request_id']}"

    text_body = f"""Hi {lead['name']},

Thank you for your interest in NETSOL.

Your request has been received successfully. A NETSOL representative will
follow up with you soon.

Request ID: {lead['request_id']}
Service: {lead['service_name']}
Your requirement: {lead['description']}

Best regards,
NETSOL
"""

    html_body = f"""
<div style="font-family: -apple-system, Arial, sans-serif; max-width: 560px; margin: 0 auto;">
  <div style="background: linear-gradient(135deg, #1b75bb, #124d78); padding: 20px 24px; border-radius: 10px 10px 0 0;">
    <h2 style="color: #fff; margin: 0; font-size: 18px;">Request Submitted Successfully</h2>
  </div>
  <div style="border: 1px solid #e2e2e2; border-top: none; border-radius: 0 0 10px 10px; padding: 20px 24px; color: #222; font-size: 14px; line-height: 1.6;">
    <p>Hi {lead['name']},</p>
    <p>Thank you for your interest in NETSOL. Your request has been received successfully, and a NETSOL representative will follow up with you soon.</p>
    <table style="width: 100%; border-collapse: collapse; font-size: 14px; margin: 16px 0;">
      <tr><td style="padding: 6px 0; color: #667; width: 140px;">Request ID</td><td style="padding: 6px 0;"><b>{lead['request_id']}</b></td></tr>
      <tr><td style="padding: 6px 0; color: #667;">Service</td><td style="padding: 6px 0;">{lead['service_name']}</td></tr>
    </table>
    <p style="color: #667; margin: 16px 0 4px 0; font-size: 13px;">Your requirement</p>
    <p style="background: #f5f7fa; border-radius: 8px; padding: 12px 14px; white-space: pre-wrap;">{lead['description']}</p>
    <p style="margin-top: 20px;">Best regards,<br/>NETSOL</p>
  </div>
</div>
"""

    return _send_email(lead["email"], subject, html_body, text_body)

