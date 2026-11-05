"""Manual runner smoke: publish only validated configuration and numeric usage."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
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
from agy_permission_probe import ProbeFailure, run_permission_probe  # noqa: E402


def template_smoke(
    *, private: Path, agy: Path, config_file: Path, timeout: float
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="workspace-", dir=private) as directory:
        work = Path(directory)
        prompt = private / "prompt.md"
        relative_template = Path(".github/ISSUE_TEMPLATE/06_parent.md")
        template = (ROOT / relative_template).read_bytes()
        (work / relative_template).parent.mkdir(parents=True)
        (work / relative_template).write_bytes(template)
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
            agy=agy,
            cwd=work,
            prompt_file=prompt,
            json_schema=schema,
            log_file=private / "agy.log",
            timeout=timeout,
            config_file=config_file,
        )
        expected = re.findall(
            r"^## (.+)$",
            template.decode("utf-8"),
            re.MULTILINE,
        )
        if result.structured_output["headings"] != expected:
            raise AgyError(
                "invalid_output",
                "smoke headings differ from the checked-out template",
                denied_actions=result.denied_actions,
                command_summary=result.command_summary,
            )
        return {
            "status": "passed",
            "observed_version": result.observed_version,
            "observed_model": result.observed_model,
            "usage": result.usage,
            "duration_seconds": result.duration_seconds,
            "elapsed_seconds": result.elapsed_seconds,
            "denied_action_count": len(result.denied_actions),
            "commands": result.command_summary,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agy", type=Path, default=DEFAULT_AGY)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--permissions-probe", action="store_true")
    args = parser.parse_args()
    report: dict[str, Any]
    state = (
        Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        / "tennis-lab-agents/agy-smoke"
    )
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    private = Path(tempfile.mkdtemp(prefix="run-", dir=state))
    try:
        if args.permissions_probe:
            report = run_permission_probe(
                private=private,
                agy=args.agy,
                config_file=args.config_file,
                timeout=args.timeout,
            )
        else:
            report = template_smoke(
                private=private,
                agy=args.agy,
                config_file=args.config_file,
                timeout=args.timeout,
            )
    except AgyError as exc:
        failure = private / "failure.json"
        with failure.open("x", encoding="utf-8") as handle:
            failure.chmod(0o600)
            json.dump(
                {"reason": exc.reason, "denied_actions": exc.denied_actions}, handle
            )
        # Error messages/logs may contain prompts, credentials or session IDs.
        report = {
            "status": "failed",
            "reason": exc.reason,
            "denied_action_count": len(exc.denied_actions),
            "commands": exc.command_summary,
        }
    except ProbeFailure as exc:
        report = {
            "status": "failed",
            "stage": "permissions",
            "reason": exc.code,
            "failed_probe": exc.probe,
        }
    except (OSError, subprocess.SubprocessError):
        report = {
            "status": "failed",
            "stage": "permissions",
            "reason": "probe_setup_failed",
        }
    finally:
        # Keep private logs/captures on every outcome; only model work is disposable.
        for name in ("work", "outside"):
            try:
                if (private / name).exists():
                    shutil.rmtree(private / name)
            except OSError:
                report = {
                    "status": "failed",
                    "stage": "permissions",
                    "reason": "probe_cleanup_failed",
                }
    return publish(report)


def publish(report: dict[str, Any]) -> int:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(f"## AGY smoke\n\n```json\n{rendered}\n```\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
