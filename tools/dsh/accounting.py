"""Account for disjoint observed token buckets; never claim a billed total."""

from __future__ import annotations

from typing import Any

TOKEN_FIELDS = (
    "inputTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
    "outputTokens",
    "totalTokens",
    "reasoningTokens",
)
# Official token-price snapshot shared by smoke and the general runner.
PRICE: dict[str, Any] = {
    "source": "https://api-docs.deepseek.com/quick_start/pricing/",
    "checked_date": "2026-10-19",
    "currency": "USD",
    "model": "deepseek-flash",
    "per_million_tokens": {
        "off_peak": {
            "inputTokens": 0.15,
            "cacheReadTokens": 0.003,
            "outputTokens": 0.6,
        },
        "peak": {"inputTokens": 0.30, "cacheReadTokens": 0.006, "outputTokens": 1.2},
    },
    "formula": "(inputTokens * input_miss_rate + cacheReadTokens * cache_hit_rate + outputTokens * output_rate) / 1000000",
    "peak_hours_utc": "Monday-Friday 01:00-04:00 and 06:00-10:00, excluding Chinese public holidays.",
    "time_basis": "Range across off-peak/peak; request billing time and Chinese public holidays are not resolved.",
}


def usage_record(
    events: list[dict[str, Any]],
    *,
    scope: str = "Fresh smoke session; step_end usage for this invocation only. Token buckets are disjoint.",
    stream_complete: bool = True,
) -> dict[str, Any]:
    steps = [
        {
            "turn": event.get("turn"),
            "step": event.get("step"),
            "usage": event.get("usage"),
        }
        for event in events
        if event.get("type") == "status" and event.get("phase") == "step_end"
    ]
    totals: dict[str, int | None] = {}
    for field in TOKEN_FIELDS:
        samples = [
            step["usage"].get(field) if isinstance(step["usage"], dict) else None
            for step in steps
        ]
        known = [n for n in samples if type(n) is int and n >= 0]
        totals[field] = sum(known) if samples and len(known) == len(samples) else None
    missing = [field for field in TOKEN_FIELDS[:4] if totals[field] is None]
    unaccounted = [
        "Auxiliary requests (session titles, compaction, etc.) may be absent from headless NDJSON.",
        "Search/Fetch and other non-token fees are not counted.",
        "This is an observed-token estimate, not the billed amount; price changes after checked_date are not tracked.",
    ]
    estimated_range = None
    if missing:
        unaccounted.append("Missing token buckets: " + ", ".join(missing))
    elif totals["cacheWriteTokens"] != 0:
        unaccounted.append("Nonzero cacheWriteTokens has no separately verified price.")
    else:
        billable = {
            field: count
            for field in ("inputTokens", "cacheReadTokens", "outputTokens")
            if (count := totals[field]) is not None
        }
        estimated_range = {
            period: sum(billable[field] * rate for field, rate in rates.items())
            / 1_000_000
            for period, rates in PRICE["per_million_tokens"].items()
        }
    if not stream_complete:
        unaccounted.append(
            "The event/step interval is incomplete; totals cover only observed complete step_end events."
        )
    return {
        "scope": scope,
        "stream_complete": stream_complete,
        "usage_complete": stream_complete and not missing,
        "steps": steps,
        "totals": totals,
        "missing_billable_buckets": missing,
        "observed_token_estimate_range_usd": estimated_range,
        "billing_total_usd": None,
        "complete": False,
        "pricing": PRICE,
        "unaccounted": unaccounted,
    }
