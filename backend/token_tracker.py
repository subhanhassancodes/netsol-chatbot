"""
token_tracker.py

Tracks Gemini token usage and estimated cost across the life of the running
server process.

How it's used:
- llm_client.generate_with_retry() calls log_usage(model, input_tokens,
  output_tokens) after every successful Gemini call.
- main.py wraps each /chat request in `with token_tracker.track_request()`.
  A single question can trigger several Gemini calls under the hood
  (routing, SQL/pandas code generation, retries, final synthesis) — all of
  them are added together into ONE summary for that request, which is
  printed to the terminal once and returned to the frontend.
- cumulative() exposes the running totals since the server started, for the
  GET /usage endpoint.

Pricing is USD per 1,000,000 tokens, at Google's list rates for the model
in use (checked July 2026): https://ai.google.dev/gemini-api/docs/pricing
"""

import contextvars
import threading
import time
from typing import Optional

PRICING = {
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}
_DEFAULT_RATES = {"input": 0.0, "output": 0.0}

_lock = threading.Lock()
_request_count = 0
_running_tokens = 0
_running_cost = 0.0

_current: "contextvars.ContextVar[Optional[_Usage]]" = contextvars.ContextVar(
    "token_tracker_current_usage", default=None
)


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING.get(model, _DEFAULT_RATES)
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


class _Usage:
    __slots__ = ("model", "input_tokens", "output_tokens", "calls", "start") 

    def __init__(self) -> None:
        self.model: Optional[str] = None 
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.start = time.perf_counter()


class track_request:
    """Context manager for one user-facing request (e.g. one POST /chat
    call). Every Gemini call made anywhere inside the block — however many
    there are — gets added into a single usage summary for that request.

    Usage:
        with token_tracker.track_request() as tracked:
            ...do work that may call generate_with_retry() any number of times...
            usage = tracked.summary()
    """

    def __enter__(self) -> "track_request":
        self._usage = _Usage()
        self._token = _current.set(self._usage)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _current.reset(self._token)

    def summary(self) -> dict:
        return _finish(self._usage)


def log_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    """Called by llm_client right after every raw Gemini response. Adds to
    whichever track_request() block is currently active, if any."""
    usage = _current.get()
    if usage is not None:
        usage.model = model
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.calls += 1


def _finish(usage: _Usage) -> dict:
    global _request_count, _running_tokens, _running_cost

    elapsed_ms = round((time.perf_counter() - usage.start) * 1000)

    if usage.calls == 0:
        # No Gemini call happened for this request (served from cache, or
        # an out-of-scope question that short-circuits before calling the
        # model) — nothing to log or charge for.
        with _lock:
            running_tokens, running_cost = _running_tokens, _running_cost
        return {
            "request_number": None,
            "model": None,
            "llm_calls": 0,
            "response_time_ms": elapsed_ms,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "running_total_tokens": running_tokens,
            "running_total_cost": round(running_cost, 6),
        }

    model = usage.model or "unknown"
    total_tokens = usage.input_tokens + usage.output_tokens
    cost = _cost(model, usage.input_tokens, usage.output_tokens)

    with _lock:
        _request_count += 1
        _running_tokens += total_tokens
        _running_cost += cost
        snapshot = {
            "request_number": _request_count,
            "model": model,
            "llm_calls": usage.calls,
            "response_time_ms": elapsed_ms,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": round(cost, 6),
            "running_total_tokens": _running_tokens,
            "running_total_cost": round(_running_cost, 6),
        }

    print("-" * 40)
    print(f"Request #{snapshot['request_number']}")
    print(f"Model: {model}")
    print(f"Response Time: {elapsed_ms:,} ms")
    print(f"Input Tokens: {usage.input_tokens:,}")
    print(f"Output Tokens: {usage.output_tokens:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print(f"Estimated Cost: ${cost:.4f}")
    print(f"Running Total Tokens: {snapshot['running_total_tokens']:,}")
    print(f"Running Total Cost: ${snapshot['running_total_cost']:.4f}")
    print("-" * 40)

    return snapshot


def cumulative() -> dict:
    """Running totals since the server process started — used by GET /usage."""
    with _lock:
        return {
            "request_count": _request_count,
            "running_total_tokens": _running_tokens,
            "running_total_cost": round(_running_cost, 6),
        }
    
    