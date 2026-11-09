"""Allow schema-checked structured completion; deny every operational tool."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from agent_events import parse_json  # noqa: E402
from agent_output import OutputSchema  # noqa: E402


def main() -> None:
    raw = sys.stdin.read()
    decision = "deny"
    reason = (
        "All input is supplied in the prompt. Do not run commands, research, or edit files. "
        "Return the requested result through the schema-checked finish tool."
    )
    try:
        payload = parse_json(raw)
        tool = payload["toolCall"]
        if tool["name"] == "finish":
            path = Path(sys.argv[2])
            if (
                path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != sys.argv[3]
            ):
                raise ValueError("schema changed")
            OutputSchema(path, agent="agy").validate(tool["args"])
            decision = "allow"
            reason = "Schema-checked structured completion only; no external operation."
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        pass
    try:
        descriptor = os.open(sys.argv[1], os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"input": raw, "decision": decision}) + "\n")
    except (OSError, IndexError):
        decision = "deny"
    print(json.dumps({"decision": decision, "reason": reason}))


if __name__ == "__main__":
    main()
