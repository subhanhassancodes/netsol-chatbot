"""
ingest.py

Standalone ingestion script for the NETSOL chatbot project.

What this does:
1. Scrapes www.netsolpk.com live (homepage + internal pages), capturing:
   - main body text per page (nav/header/footer stripped out, as before)
   - the site's footer/navigation text ONCE, as its own separate chunk
     (this is where things like "Products", "Services", "Contact Us",
     office address, and phone numbers live — useful content that was
     previously being discarded)
2. Chunks all of that text and generates embeddings
3. Stores everything in a persistent ChromaDB collection (./chroma_db)
4. Loads the employee JSON export (zeropaper_staging_employee_profiles.json),
   normalizes every field it contains — scalar and list alike — into
   relational tables (see build_sql_store() for how)
5. Writes those tables into a SQLite database (netsol_hr.db)

Run this once to build your data stores. Re-run it any time you want to
refresh the website content or the employee JSON changes.
"""

import json
import os
import sqlite3
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import chromadb
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


# ---------- Config ----------
BASE_URL = "https://www.netsolpk.com/"
ALLOWED_DOMAIN = "www.netsolpk.com"
# The employee CSV export has been retired in favor of a direct JSON export
# (a nested, MongoDB-shaped document per employee — see load_employee_records()
# and build_sql_store() below). This is now the single source of truth for
# employee data; there is no CSV fallback.
EMPLOYEE_JSON_PATH = "zeropaper_staging_employee_profiles.json"
SCRAPED_JSON_PATH = "netsol_scraped.json"  # written fresh each run, kept as a local record
CHROMA_DB_PATH = "./chroma_db"
SQLITE_DB_PATH = "netsol_hr.db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Pages linked from the footer that live on OTHER NETSOL domains
# (netsoltech.com, ir.netsoltech.com, appexnow.com, otozmobility.com).
# These aren't crawled automatically since they're different sites with
# their own structure — we scrape this specific, known list instead.
EXTRA_URLS = [
    "https://netsoltech.com/about-us/csr",
    "https://netsoltech.com/products/transcend",
    "https://netsoltech.com/products/digital-retail",
    "https://netsoltech.com/products/portals",
    "https://netsoltech.com/products/loan-origination",
    "https://netsoltech.com/products/lease-servicing",
    "https://netsoltech.com/products/wholesale-finance",
    "https://netsoltech.com/products/mobility-solutions",
    "https://netsoltech.com/services/information-security",
    "https://netsoltech.com/services/digital-solutions",
    "https://netsoltech.com/services/ai-ml-services-and-solutions",
    "https://netsoltech.com/services/generative-ai",
    "https://netsoltech.com/services/emerging-technologies",
    "https://netsoltech.com/services/cloud-services",
    "https://netsoltech.com/services/data-engineering",
    "https://netsoltech.com/solutions/asset-finance",
    "https://netsoltech.com/solutions/automotive-finance",
    "https://netsoltech.com/solutions/equipment-finance",
    "https://ir.netsoltech.com/company-information",
    "https://ir.netsoltech.com/press-releases",
    "https://ir.netsoltech.com/stock-data",
    "https://otozmobility.com/",
]


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _clean_text(soup_fragment) -> str:
    text = soup_fragment.get_text(separator="\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def scrape_site() -> dict:
    """Scrapes the site and returns {url_or_label: text}, including one
    special 'site-footer-navigation' entry captured from the homepage."""

    print(f"Fetching homepage: {BASE_URL}")
    home_response = requests.get(BASE_URL, headers=HEADERS, timeout=10)
    fresh_soup = BeautifulSoup(home_response.text, "html.parser")

    # Discover internal links (same approach as before: relative + absolute,
    # restricted to www.netsolpk.com, skipping anchors/js/pdf links).
    links = {BASE_URL}
    for a in fresh_soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#") or href.startswith("javascript:"):
            continue
        full_url = urljoin(BASE_URL, href)
        if urlparse(full_url).netloc != ALLOWED_DOMAIN:
            continue
        if full_url.lower().endswith((".pdf", ".doc", ".docx", ".xlsx")):
            continue
        links.add(full_url.split("#")[0])

    print(f"Found {len(links)} pages to scrape")

    # Capture the footer ONCE (it's the same across pages) before we start
    # stripping it out of each page's body text.
    footer_tag = fresh_soup.find("footer")
    all_pages_text = {}
    if footer_tag:
        footer_text = _clean_text(footer_tag)
        if footer_text:
            all_pages_text["site-footer-navigation"] = footer_text
            print(f"Captured footer/navigation content ({len(footer_text)} chars)")

    # Now scrape each page's main body text, same cleaning approach as before.
    for link in sorted(links):
        try:
            r = requests.get(link, headers=HEADERS, timeout=10)
            page_soup = BeautifulSoup(r.text, "html.parser")

            for tag in page_soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            clean_text = _clean_text(page_soup)
            all_pages_text[link] = clean_text
            print(f"Scraped: {link} ({len(clean_text)} chars)")
        except Exception as e:
            print(f"Failed: {link} — {e}")

    # Scrape the additional pages on other NETSOL domains (products, services,
    # solutions, investor relations, etc.) linked from the footer.
    print(f"\nScraping {len(EXTRA_URLS)} additional pages from other NETSOL domains...")
    for link in EXTRA_URLS:
        try:
            r = requests.get(link, headers=HEADERS, timeout=10)
            page_soup = BeautifulSoup(r.text, "html.parser")

            for tag in page_soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            clean_text = _clean_text(page_soup)
            if clean_text:
                all_pages_text[link] = clean_text
                print(f"Scraped: {link} ({len(clean_text)} chars)")
            else:
                print(f"Skipped (empty after cleaning): {link}")
        except Exception as e:
            print(f"Failed: {link} — {e}")

    return all_pages_text


def build_rag_store():
    print("=== Building RAG store (ChromaDB) ===")

    all_pages_text = scrape_site()

    # Save a local copy for reference / debugging, same as before.
    with open(SCRAPED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_pages_text, f, ensure_ascii=False, indent=2)
    print(f"Saved raw scraped text to {SCRAPED_JSON_PATH}")

    all_chunks = []
    for source, text in all_pages_text.items():
        if source == "site-footer-navigation":
            domain = ALLOWED_DOMAIN
        else:
            domain = urlparse(source).netloc or ALLOWED_DOMAIN
        for chunk in chunk_text(text):
            all_chunks.append({"text": chunk, "source": source, "domain": domain})
    print(f"Created {len(all_chunks)} chunks")

    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in all_chunks]
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"Writing to persistent ChromaDB at {CHROMA_DB_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        client.delete_collection("netsol_site")
    except Exception:
        pass

    collection = client.create_collection("netsol_site")
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": c["source"], "domain": c["domain"]} for c in all_chunks],
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
    )
    print(f"Added {collection.count()} documents to ChromaDB\n")


def load_employee_records() -> list[dict]:
    """Loads the raw employee JSON as a list of per-employee dicts (NOT
    flattened — build_sql_store() below needs the original nested
    lists/dicts intact to normalize them generically). Accepts a plain
    JSON array of records (what we actually get), or an object containing
    one under a key like "data"/"records"/"rows"/"employees"/"items"."""
    if not os.path.exists(EMPLOYEE_JSON_PATH):
        raise FileNotFoundError(f"Employee data file not found: {EMPLOYEE_JSON_PATH}")

    with open(EMPLOYEE_JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        records = None
        for key in ("data", "records", "rows", "employees", "items"):
            if isinstance(raw.get(key), list):
                records = raw[key]
                break
        if records is None:
            records = [raw]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("Employee JSON must be a list of records or an object containing one.")

    return records


def _is_scalar(value) -> bool:
    return not isinstance(value, (list, dict))


def build_sql_store():
    """Builds netsol_hr.db from the employee JSON export.

    The export is a nested, MongoDB-shaped document per employee: most
    fields (name, salary, designation, ...) are plain scalars, but things
    like education, job_experience, skills, languages,
    tools_or_technologies, certifications, referees, and every "_history"
    field (designation/department/team/grade/promotion) are lists living
    directly on the record — there's no pre-flattened "skills[0]" style
    column to key off of like the old CSV export had.

    Rather than hardcoding which of those fields exist (the old version
    did, via a fixed employee_columns list and a fixed set of per-field
    loops with a hardcoded max-index guess for each) — which is also
    exactly why it silently dropped several fields the new JSON has
    (languages, tools_or_technologies, certifications, referees, and all
    the *_history fields) and produced wrong/incomplete answers — this
    walks every record generically:
      - any field that's a scalar (str/int/float/bool/None) on every
        record it appears on goes into one flat `employees` table
      - any field that holds a LIST anywhere in the dataset gets its own
        table, named after the field, with an `employee_id` foreign key:
        a list of dicts (e.g. education) becomes one row per dict, with
        the dict's own keys as columns; a list of plain values (e.g.
        skills) becomes one row per value, in a `value` column
      - `_id` (Mongo's internal document id) is dropped — it's not
        meaningful data
    A field that's genuinely empty for every employee in the current
    export (e.g. "training") has nothing to build a table from and is
    skipped for this run; it'll appear automatically the next time
    ingest.py runs against an export that actually has data in it.

    This means a field added to the JSON later — new or renamed, scalar
    or list — is picked up automatically the next time this script runs,
    with no code changes needed here or in chatbot.py (its Text2SQL
    prompt already reads the schema straight from sqlite_master, so new
    tables/columns just show up)."""
    print("=== Building SQL store (SQLite) from employee JSON ===")

    records = load_employee_records()
    print(f"Loaded employee data from {EMPLOYEE_JSON_PATH}: {len(records)} records")

    list_fields = set()
    for rec in records:
        for key, value in rec.items():
            if isinstance(value, list):
                list_fields.add(key)

    # ---- employees: one row per record, every scalar top-level field ----
    employee_rows = [
        {k: v for k, v in rec.items() if k != "_id" and k not in list_fields and _is_scalar(v)}
        for rec in records
    ]
    employees = pd.DataFrame(employee_rows)
    if "employee_id" in employees.columns:
        employees = employees[["employee_id"] + [c for c in employees.columns if c != "employee_id"]]

    conn = sqlite3.connect(SQLITE_DB_PATH)
    employees.to_sql("employees", conn, if_exists="replace", index=False)
    print(f"employees: {employees.shape}")

    # Any table from a previous run (old CSV-derived schema, or a field
    # that no longer appears in this export) that we're not about to
    # rewrite below would otherwise linger as stale, disconnected data —
    # drop every non-employees table first so the database always
    # reflects exactly this run's JSON, nothing else.
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'employees'")
    for (old_table,) in cur.fetchall():
        cur.execute(f'DROP TABLE IF EXISTS "{old_table}"')

    # ---- one table per list field ----
    for field in sorted(list_fields):
        sub_rows = []
        for rec in records:
            employee_id = rec.get("employee_id")
            for item in (rec.get(field) or []):
                if isinstance(item, dict):
                    sub_rows.append({"employee_id": employee_id, **item})
                else:
                    sub_rows.append({"employee_id": employee_id, "value": item})
        if not sub_rows:
            print(f"{field}: 0 rows (empty for every employee) — table skipped")
            continue
        table = pd.DataFrame(sub_rows)
        table.to_sql(field, conn, if_exists="replace", index=False)
        print(f"{field}: {table.shape}")

    conn.commit()
    conn.close()

    print(f"Wrote SQLite database to {SQLITE_DB_PATH}\n")


if __name__ == "__main__":
    build_rag_store()
    build_sql_store()
    print("=== Ingestion complete ===")

    