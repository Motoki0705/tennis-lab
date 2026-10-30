"""Manual runner smoke: publish only validated configuration and numeric usage."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".agents/ops/lib"))

from agy_headless import (  # noqa: E402 - standalone script bootstraps the ops imports
    DEFAULT_AGY,
    DEFAULT_CONFIG,
    AgyError,
    run_headless,
)


def safe_denials(actions: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            key: value if re.fullmatch(r"[A-Za-z0-9_:-]{1,80}", value) else "[redacted]"
            for key in ("action", "display_name")
            if (value := action.get(key)) is not None
        }
        for action in actions
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agy", type=Path, default=DEFAULT_AGY)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    report: dict[str, Any]
    try:
        with tempfile.TemporaryDirectory(prefix="agy-smoke-") as directory:
            private = Path(directory)
            prompt = private / "prompt.md"
            prompt.write_text(
                "Read .github/ISSUE_TEMPLATE/06_parent.md in this checkout. "
                "Return all level-two headings, in order, without the '## ' prefix. "
                "Use only read operations and return the requested JSON schema.",
                encoding="utf-8",
            )
            schema = private / "schema.json"
            schema.write_text(
                json.dumps(
                    {
                        "type": "object",
                        "properties": {
                            "headings": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["headings"],
                        "additionalProperties": False,
                    }
                ),
                encoding="utf-8",
            )
            result = run_headless(
                agy=args.agy,
                cwd=ROOT,
                prompt_file=prompt,
                json_schema=schema,
                log_file=private / "agy.log",
                timeout=args.timeout,
                config_file=args.config_file,
            )
            expected = re.findall(
                r"^## (.+)$",
                (ROOT / ".github/ISSUE_TEMPLATE/06_parent.md").read_text(
                    encoding="utf-8"
                ),
                re.MULTILINE,
            )
            if result.structured_output["headings"] != expected:
                raise AgyError(
                    "invalid_output",
                    "smoke headings differ from the checked-out template",
                    denied_actions=result.denied_actions,
                )
            report = {
                "status": "passed",
                "observed_version": result.observed_version,
                "observed_model": result.observed_model,
                "usage": result.usage,
                "duration_seconds": result.duration_seconds,
                "elapsed_seconds": result.elapsed_seconds,
                "denied_actions": safe_denials(result.denied_actions),
            }
    except AgyError as exc:
        # Error messages/logs may contain prompts, credentials or session IDs.
        report = {
            "status": "failed",
            "reason": exc.reason,
            "denied_actions": safe_denials(exc.denied_actions),
        }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(f"## AGY smoke\n\n```json\n{rendered}\n```\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
