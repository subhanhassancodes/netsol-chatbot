"""
document_store.py

Everything about an uploaded document lives here: storing it per session,
and answering questions about it.

Two kinds of upload are supported, keyed by "kind" in the stored entry:
- "text" — .txt / .pdf. The extracted text is handed straight to the LLM as
  context (chatbot.py's document_text path handles the actual synthesis).
- "csv"  — .csv OR .json. Both parse into the same pandas DataFrame and
  are answered with real computation: gather_csv_answer() asks Gemini to
  translate the question into one line of pandas code, runs it in a
  sandboxed eval() against the actual data, and returns the real computed
  result — not a guess. This is what makes these questions reliable
  instead of the LLM eyeballing a wall of text and getting numbers wrong.
  (The stored "kind" stays "csv" either way — it describes the code path,
  not the original file extension.)

Design notes:
- Storage is a simple dict keyed by session_id, living only in this server
  process's memory. Restarting the server clears everything. Nothing is
  written to disk, shared across sessions, or persisted anywhere.
- Text extraction is a small registry (_TEXT_EXTRACTORS) keyed by file
  extension — add a new text file type by writing one function and
  registering it, without touching any other code.
- Currently supports: .txt, .pdf (text) and .csv, .json (tabular data).
"""

import io
import json
import re
from typing import Dict, Optional

import pandas as pd

from backend.llm_client import generate_with_retry

MAX_CHARS = 15000  # keep extracted text at a reasonable size for the LLM's context
MAX_ROWS_SHOWN = 20  # cap on rows/values shown back from a CSV computation

# session_id -> {"kind": "text", "filename": str, "text": str}
#            or {"kind": "csv",  "filename": str, "df": pd.DataFrame}
_session_documents: Dict[str, dict] = {}


# ==================== Storage ====================

def _extract_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def _extract_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader  # imported lazily so txt/csv-only usage doesn't need pypdf installed
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# Add new text file types here: extension -> function(file_bytes) -> extracted text
_TEXT_EXTRACTORS = {
    "txt": _extract_txt,
    "pdf": _extract_pdf,
}

# Both parse into the same DataFrame shape and go through the same
# question-answering path (gather_csv_answer) — the file extension only
# decides how the bytes get turned into rows and columns.
TABULAR_EXTENSIONS = {"csv", "json"}


def supported_extensions() -> list[str]:
    return sorted(set(_TEXT_EXTRACTORS) | TABULAR_EXTENSIONS)


def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


_DATE_NAME_HINT = re.compile(r"date|_at$|^dob$|birth|joined|hired", re.IGNORECASE)


def _infer_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Shared by both the CSV and JSON parsers. Converts columns that are
    both named like a date (contains "date", "_at", "dob", "birth",
    "joined", "hired") AND whose non-null values actually parse as dates,
    into real datetime64 columns.

    Both conditions matter: name-only would wrongly convert nothing since
    plenty of date-named columns come in already as proper types from
    JSON; content-only was what previously misread zero-padded IDs like
    employee_id ('0001', '0002', ...) as valid (if nonsensical) dates. The
    success-rate check is measured against non-null values only, so a
    sparsely-populated but genuine date column (e.g. termination_date,
    mostly blank because most employees haven't left) isn't penalized for
    its missing values and skipped."""
    for col in df.columns:
        if not (pd.api.types.is_string_dtype(df[col]) and _DATE_NAME_HINT.search(str(col))):
            continue
        non_null = df[col].notna()
        if non_null.sum() == 0:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        success_rate = parsed[non_null].notna().mean()
        if success_rate > 0.8:
            df[col] = parsed
    return df


def _parse_csv(file_bytes: bytes) -> pd.DataFrame:
    """Parses the CSV, auto-detecting headers and inferring data types.
    pandas already infers numbers/booleans; _infer_dates adds a best-effort
    pass to also catch date columns, which read_csv otherwise leaves as
    plain strings. Missing values are left as NaN (handled gracefully
    downstream by pandas' own NaN-aware aggregations) rather than dropped,
    so rows with a few blank fields still count everywhere else."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    return _infer_dates(df)


def _parse_json(file_bytes: bytes) -> pd.DataFrame:
    """Parses a JSON file into the same flat row/column shape as the CSV
    path. Accepts a JSON array of objects (the common "list of records"
    shape — e.g. [{"employee_name": "...", "salary_amount": 50000}, ...])
    or a single object containing such an array under a key like "data",
    "records", "rows", or "employees". Nested objects (e.g. {"education":
    {"degree": "..."}}) are flattened into dot-separated columns
    (education.degree) via pandas' json_normalize, mirroring how the CSV's
    bracket-indexed columns (education[0].degree) already work."""
    parsed = json.loads(file_bytes.decode("utf-8"))

    if isinstance(parsed, dict):
        records = None
        for key in ("data", "records", "rows", "employees", "items"):
            if isinstance(parsed.get(key), list):
                records = parsed[key]
                break
        if records is None:
            # A single JSON object with no obvious list-of-records key —
            # treat it as one record.
            records = [parsed]
    elif isinstance(parsed, list):
        records = parsed
    else:
        raise ValueError("Expected a JSON array of records, or an object containing one.")

    df = pd.json_normalize(records)
    df.columns = [str(c).strip() for c in df.columns]
    return _infer_dates(df)


def store_document(session_id: str, filename: str, file_bytes: bytes) -> dict:
    """Extracts/parses the uploaded file and stores it for this session
    only. Overwrites anything previously uploaded (text or tabular data)
    for the same session. Returns a small metadata dict describing what
    was stored, shaped differently depending on kind:
    - csv:  {"kind": "csv", "filename": ..., "rows": ..., "columns": ...}
    - text: {"kind": "text", "filename": ..., "chars": ...}
    """
    ext = _extension(filename)

    if ext in TABULAR_EXTENSIONS:
        try:
            df = _parse_csv(file_bytes) if ext == "csv" else _parse_json(file_bytes)
        except Exception as e:
            raise ValueError(f"Couldn't parse that {ext.upper()} file: {e}")
        if df.empty:
            raise ValueError(f"That {ext.upper()} file has no rows.")

        _session_documents[session_id] = {"kind": "csv", "filename": filename, "df": df}
        return {"kind": "csv", "filename": filename, "rows": len(df), "columns": len(df.columns)}

    extractor = _TEXT_EXTRACTORS.get(ext)
    if not extractor:
        raise ValueError(
            f"Unsupported file type: .{ext}. Supported types: "
            f"{', '.join('.' + e for e in supported_extensions())}"
        )

    text = extractor(file_bytes).strip()
    if not text:
        raise ValueError("No extractable text was found in that file.")
    text = text[:MAX_CHARS]

    _session_documents[session_id] = {"kind": "text", "filename": filename, "text": text}
    return {"kind": "text", "filename": filename, "chars": len(text)}


def has_document(session_id: str) -> bool:
    return session_id in _session_documents


def has_csv(session_id: str) -> bool:
    doc = _session_documents.get(session_id)
    return bool(doc) and doc.get("kind") == "csv"


def get_dataframe(session_id: str) -> Optional[pd.DataFrame]:
    doc = _session_documents.get(session_id)
    return doc["df"] if doc and doc.get("kind") == "csv" else None


def get_document_text(session_id: str) -> Optional[str]:
    doc = _session_documents.get(session_id)
    return doc["text"] if doc and doc.get("kind") == "text" else None


def get_document_filename(session_id: str) -> Optional[str]:
    doc = _session_documents.get(session_id)
    return doc["filename"] if doc else None


def clear_document(session_id: str) -> None:
    _session_documents.pop(session_id, None)


# ==================== CSV question answering ====================
# Mirrors the Text2SQL approach used elsewhere in this project: the LLM's
# only job is to translate the question into pandas code; a sandboxed
# eval() actually runs it against the real dataframe, so the answer is
# always the real computed result, never a guess.

def _is_nested_column(series: pd.Series) -> bool:
    """True if this column holds Python list/dict objects — the shape JSON
    naturally produces for nested data (e.g. education, skills,
    designation_history) when it ISN'T pre-flattened into bracket-indexed
    columns like a CSV export would be. These can't be rendered as a
    normal scalar value, and dumping their raw repr into the sample table
    is exactly what was drowning the prompt in noise."""
    if series.dtype != object:
        return False
    for value in series.dropna().head(20):
        if isinstance(value, (list, dict)):
            return True
    return False


def _profile_dataframe(df: pd.DataFrame, sample_rows: int = 5) -> str:
    """A compact text profile of the dataframe — columns, inferred types,
    missing-value counts, and a few sample rows. Enough for the LLM to
    write correct pandas code without guessing column names or types.

    Two different "too much data" problems are handled here:
    - Wide CSVs with repeated indexed columns (e.g. skills[0], skills[1],
      ..., skills[50]) are collapsed into one summary line each instead of
      dozens of near-duplicate lines.
    - JSON sources that keep the SAME data as genuinely nested lists/dicts
      in a single column (e.g. designation_history, education, skills) are
      excluded from the sample-rows dump entirely — rendering 5 rows of
      raw nested Python dict reprs was bloating the prompt to 3000+ tokens
      of noise per question and very plausibly degrading the model's
      ability to correctly use simple flat columns like gender that were
      sitting right next to all that clutter."""
    indexed_pattern = re.compile(r"^(.*)\[(\d+)\](.*)$")
    groups: "dict[tuple, list[int]]" = {}
    scalar_cols = []
    nested_cols = []

    for col in df.columns:
        m = indexed_pattern.match(str(col))
        if m:
            prefix, idx, suffix = m.group(1), int(m.group(2)), m.group(3)
            groups.setdefault((prefix, suffix), []).append(idx)
        elif _is_nested_column(df[col]):
            nested_cols.append(col)
        else:
            scalar_cols.append(col)

    lines = [f"Shape: {len(df):,} rows x {len(df.columns)} columns", "", "Columns:"]

    for col in scalar_cols:
        dtype = str(df[col].dtype)
        n_missing = int(df[col].isna().sum())
        missing_note = f", {n_missing} missing" if n_missing else ""
        lines.append(f"  - {col!r} ({dtype}{missing_note})")

    if groups:
        lines.append("")
        lines.append("Repeated/indexed columns (grouped — use the exact bracketed name, e.g. skills[3], for a specific index):")
        for (prefix, suffix), indices in sorted(groups.items()):
            indices.sort()
            example_col = f"{prefix}[{indices[0]}]{suffix}"
            dtype = str(df[example_col].dtype)
            lines.append(f"  - {prefix}[0..{indices[-1]}]{suffix} — {len(indices)} columns ({dtype})")

    if nested_cols:
        lines.append("")
        lines.append("Nested columns (each cell holds a list/dict of sub-records, not a plain value — avoid these unless the question specifically asks about their contents; if you must use one, write list/dict comprehension code, e.g. `df['education'].apply(lambda x: x[0]['degree'] if x else None)`):")
        for col in nested_cols:
            n_missing = int(df[col].isna().sum())
            missing_note = f", {n_missing} missing" if n_missing else ""
            lines.append(f"  - {col!r}{missing_note}")

    lines.append("")
    lines.append(f"Sample values (first {sample_rows} rows, scalar columns only — nested columns omitted to keep this readable):")
    sample_cols = scalar_cols if scalar_cols else list(df.columns)
    lines.append(df[sample_cols].head(sample_rows).to_string())

    return "\n".join(lines)


def _question_to_pandas(
    question: str,
    profile: str,
    previous_error: str = None,
    previous_code: str = None,
    history_context: str = "",
) -> str:
    error_context = ""
    if previous_error:
        error_context = f"""

Your previous attempt failed. Fix it.
Previous code: {previous_code}
Error: {previous_error}"""

    history_block = f"\n{history_context}\n" if history_context else ""

    prompt = f"""You are a pandas expert. A dataframe called df is already loaded in memory.

{profile}
{history_block}
Question: {question}

Rules:
- Return ONLY a single Python expression that evaluates to the answer. No explanation, no markdown, no backticks, no assignment (=), no print(), no import statements, no semicolons.
- Use only the variable `df` and the `pd` module (pandas is already imported as pd).
- Match column names EXACTLY as shown above, including case and brackets (e.g. `skills[0]`, not `skills_0`).
- For matching text/category values, prefer `.str.contains(value, case=False, na=False)` over exact `==` equality, unless the question clearly wants an exact match (e.g. a specific ID).
- EXCEPTION: for the "gender" column, always use exact equality (`df['gender'] == 'Male'`), NEVER `.str.contains('Male', ...)`. The word "Male" is a substring of "Female", so a contains-match on "Male" incorrectly counts female employees too.
- For a count/average/sum/min/max/median or similar aggregate question, return that single number (e.g. `df['salary_amount'].mean()`, `len(df)`, `df['salary_amount'].max()`).
- For "how many X and Y are there" / "which X exist" / breakdown-by-category questions, return a `value_counts()` Series, e.g. `df['gender'].value_counts()` or `df['current_department'].value_counts()`.
- For "who has the highest/lowest/most/oldest/newest ___" questions, use `.loc[df[col].idxmax()]` or `.idxmin()` and select the relevant identifying + value columns, e.g. `df.loc[df['salary_amount'].idxmax(), ['employee_name', 'salary_amount']]`. For "oldest employee" use the earliest `date_of_birth`; for "most recently joined" use the latest `joining_date`.
- For "show me / list / filter" questions, return a dataframe of the relevant rows, but select only a SENSIBLE subset of identifying columns (e.g. name, department, designation, the field asked about) rather than every column — this dataframe may have 100+ columns and dumping all of them is unreadable. Cap rows with `.head({MAX_ROWS_SHOWN})` unless the question explicitly asks for all rows.
- For "sort by ___" questions, use `.sort_values('col', ascending=...)` and select a sensible subset of columns, capped with `.head({MAX_ROWS_SHOWN})`.
- For contact details / email / phone questions about a named person, filter on `employee_name` with `.str.contains(name, case=False, na=False)` and select the relevant columns (`email`, `mobile`, etc.).
- There may be no dedicated "city" column — if the question asks about city and no such column exists, the closest available field is usually `address` (free text); return that column directly rather than guessing a parsing rule, e.g. `df[['employee_name', 'address']].head({MAX_ROWS_SHOWN})`.
- For date columns, use pandas datetime comparisons directly (they're already parsed as datetimes where detected) — e.g. `df[df['joining_date'] > '2023-01-01']`.
- If the question uses a pronoun or implicit reference (e.g. "his pay", "that department", "her email"), resolve WHO or WHAT it refers to using the recent conversation above before writing the expression — do not guess or pick an arbitrary row. If there's no recent conversation to resolve it from, do your best with the question text alone.{error_context}

Expression:"""

    code = generate_with_retry(prompt).strip()
    return code.replace("```python", "").replace("```", "").strip()


# Blocks anything that could escape the sandbox or touch the outside world.
_FORBIDDEN = re.compile(
    r"\b(import|exec|eval|open|__\w+__|os\.|sys\.|subprocess|globals|locals|input|compile|getattr|setattr|delattr)\b"
)


def _safe_eval(code: str, df: pd.DataFrame):
    if _FORBIDDEN.search(code):
        raise ValueError("Generated code used a disallowed operation.")

    safe_builtins = {
        "len": len, "sum": sum, "min": min, "max": max, "round": round,
        "sorted": sorted, "abs": abs, "list": list, "dict": dict, "str": str,
        "int": int, "float": float, "bool": bool, "range": range,
    }
    return eval(code, {"__builtins__": safe_builtins, "pd": pd}, {"df": df})


def _format_result(result) -> str:
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "(No matching rows were found.)"
        shown = result.head(MAX_ROWS_SHOWN)
        note = "" if len(result) <= MAX_ROWS_SHOWN else f"\n(showing {MAX_ROWS_SHOWN} of {len(result):,} matching rows)"
        return shown.to_string(index=False) + note
    if isinstance(result, pd.Series):
        if result.empty:
            return "(No matching rows were found.)"
        shown = result.head(MAX_ROWS_SHOWN)
        note = "" if len(result) <= MAX_ROWS_SHOWN else f"\n(showing {MAX_ROWS_SHOWN} of {len(result):,} values)"
        return shown.to_string() + note
    return str(result)


def gather_csv_answer(session_id: str, question: str, history_context: str = "") -> dict:
    """Returns {"result": <formatted computed answer>, "code": <pandas code
    used>, "error": None}, or {"result": None, "code": ..., "error": <msg>}
    if both the first attempt and the one retry failed. Looks up the
    dataframe for session_id itself, so callers (chatbot.py) don't need to
    juggle the dataframe directly."""
    df = get_dataframe(session_id)
    if df is None:
        return {"result": None, "code": None, "error": "No CSV is stored for this session."}

    profile = _profile_dataframe(df)
    code = _question_to_pandas(question, profile, history_context=history_context)

    try:
        result = _safe_eval(code, df)
        return {"result": _format_result(result), "code": code, "error": None}
    except Exception as first_error:
        try:
            fixed_code = _question_to_pandas(
                question, profile, previous_error=str(first_error), previous_code=code, history_context=history_context
            )
            result = _safe_eval(fixed_code, df)
            return {"result": _format_result(result), "code": fixed_code, "error": None}
        except Exception as second_error:
            return {"result": None, "code": code, "error": str(second_error)}
        