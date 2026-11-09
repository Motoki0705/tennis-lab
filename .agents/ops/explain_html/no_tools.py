"""Trusted PreToolUse hook: explain-html turns consume only their supplied context."""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    raw = sys.stdin.read()
    try:
        descriptor = os.open(sys.argv[1], os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"input": raw, "decision": "deny"}) + "\n")
    finally:
        # Even an unavailable audit file must never turn a denial into an allow.
        print(
            json.dumps(
                {
                    "decision": "deny",
                    "reason": "All input is supplied in the prompt. Do not use tools, "
                    "research, or edit files. Return the requested structured output.",
                }
            )
        )


if __name__ == "__main__":
    main()
