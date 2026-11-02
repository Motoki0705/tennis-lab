"""Private, sequential manual validation using the production agy/pipeline code.

Only this manual controller temporarily substitutes the existing census profile
preflight. Production run_headless, workflow and smoke keep their strict guard.
Historical snapshots bypass live eligibility gates, never generation checks.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / ".agents/ops"),
    str(ROOT / ".agents/ops/lib"),
    str(Path(__file__).parent),
]
agy: Any = importlib.import_module("agy_headless")
census: Any = importlib.import_module("agy_command_census")
probe: Any = importlib.import_module("agy_permission_probe")
pipeline: Any = importlib.import_module("issue_rewrite.pipeline")
CAP = 20


class ValidationStop(RuntimeError):
    """Stop the entire campaign without retrying a quota, policy or budget error."""


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes() if path.exists() else b"{}").hexdigest()


def saved_digest() -> str:
    return digest(Path.home() / ".gemini/antigravity-cli/settings.json")


class Ledger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "invocations.jsonl"

    def rows(self) -> list[dict[str, Any]]:
        rows = (
            [json.loads(line) for line in self.path.read_text().splitlines()]
            if self.path.exists()
            else []
        )
        if [row["ordinal"] for row in rows] != list(range(1, len(rows) + 1)) or len(
            rows
        ) > CAP:
            raise ValidationStop("invalid_ledger")
        return rows

    @property
    def remaining(self) -> int:
        return CAP - len(self.rows())

    def reserve(self, **metadata: Any) -> int:
        ordinal = len(self.rows()) + 1
        if ordinal > CAP:
            raise ValidationStop("turn_budget_exhausted")
        with self.path.open("a", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(
                json.dumps(
                    {
                        **metadata,
                        "ordinal": ordinal,
                        "reserved_at": dt.datetime.now(dt.UTC).isoformat(),
                    }
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        return ordinal


class ManualPolicy:
    def __init__(self, expected_digest: str) -> None:
        self.expected = expected_digest
        self.owned: dict[Path, dict[Path, bytes]] = {}

    def check(self, cwd: Path) -> None:
        try:
            actual = census.preflight(cwd, self.owned.get(cwd.resolve()))
            if actual != self.expected:
                raise ValueError("persistent settings changed")
        except (OSError, ValueError) as exc:
            raise agy.AgyError("unsafe_user_settings", str(exc)) from exc

    @contextlib.contextmanager
    def scope(self) -> Iterator[None]:
        install = agy.install_command_hook

        def record_install(cwd: Path, record: Path) -> dict[Path, bytes]:
            owned: dict[Path, bytes] = install(cwd, record)
            self.owned[cwd.resolve()] = owned
            return owned

        # These substitutions exist only in this controller process and scope.
        # No argument, environment switch, profile file or production code changes.
        with (
            patch.object(agy, "_check_user_settings", self.check),
            patch.object(agy, "install_command_hook", record_install),
        ):
            yield


def effective_policy(capture: dict[str, Any], log: str) -> bool:
    return (
        (capture.get("init") or {}).get("permission_mode") == "request-review"
        and "applying agent mode plan" in log
        and capture.get("session_settings_integrity", {}).get("unchanged") is True
    )


def quota_or_auth_error(reason: str, detail: str, capture: dict[str, Any]) -> bool:
    # Never search issue/prompt/model prose for quota words. Only CLI exceptions
    # and terminal error envelopes are relevant.
    result = capture.get("result") or {}
    error = result.get("error", "") if isinstance(result, dict) else ""
    terminal = detail.rpartition(" exited ")[2] if reason == "cli_error" else ""
    text = terminal + " " + json.dumps(error)
    if isinstance(result, dict) and result.get("status") == "ERROR":
        text += " " + str(result.get("response", ""))
    return bool(
        re.search(
            r"quota|resource.exhausted|usage.limit|credits|unauthenticated|authentication|auth_required|sign.in|login.required",
            text,
            re.I,
        )
    )


class Runner:
    def __init__(self, root: Path, executable: Path, expected_digest: str) -> None:
        self.root = root.resolve()
        self.executable = executable.resolve()
        self.policy = ManualPolicy(expected_digest)
        self.ledger = Ledger(root)
        self.phase = "diagnostic"
        self.number: int | None = None
        self.expected_context: dict[str, Any] | None = None

    def event(self, event: str, **fields: Any) -> None:
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(
                json.dumps(
                    {
                        "event": event,
                        "at": dt.datetime.now(dt.UTC).isoformat(),
                        **fields,
                    }
                )
                + "\n"
            )
        print(json.dumps({"event": event, **fields}), flush=True)

    def turn(self, **kwargs: Any) -> Any:
        work = kwargs["cwd"].resolve()
        if not work.is_relative_to(self.root):
            raise ValidationStop("workspace_outside_private_root")
        if (
            self.phase == "diagnostic"
            and sum(row["phase"] == "diagnostic" for row in self.ledger.rows()) >= 3
        ):
            raise ValidationStop("diagnostic_budget_exhausted")
        if self.expected_context is not None and kwargs["log_file"].stem == "writer-1":
            prompt = kwargs["prompt_file"].read_text()
            actual_context = json.loads(prompt.split("\n\n入力（JSON）:\n", 1)[1])
            if actual_context != self.expected_context:
                raise ValidationStop("frozen_initial_context_changed")
        self.policy.owned.pop(work, None)
        self.policy.check(work)
        log: Path = kwargs["log_file"]
        ordinal = self.ledger.reserve(
            phase=self.phase, issue_number=self.number, role=log.stem, log=str(log)
        )
        before = saved_digest()
        canary = work / ("AGENTS.md" if (work / "AGENTS.md").exists() else "a")
        inside_before = digest(canary)
        outside = self.root / "outside-canary"
        outside_before = digest(outside)
        started = time.monotonic()
        self.event(
            "turn_start",
            ordinal=ordinal,
            phase=self.phase,
            issue_number=self.number,
            role=log.stem,
        )
        result = None
        error = None
        try:
            result = agy.run_headless(**{**kwargs, "agy": self.executable})
        except Exception as exc:
            error = exc
        finally:
            capture_path = log.with_name(log.name + ".commands.json")
            capture = (
                json.loads(capture_path.read_text()) if capture_path.exists() else {}
            )
            actual_log = log.read_text(errors="replace") if log.exists() else ""
            after = saved_digest()
            policy_ok = effective_policy(capture, actual_log)
            record = {
                "ordinal": ordinal,
                "phase": self.phase,
                "issue_number": self.number,
                "role": log.stem,
                "elapsed_seconds": time.monotonic() - started,
                "duration_seconds": result.duration_seconds
                if result is not None
                else None,
                "error": getattr(error, "reason", type(error).__name__)
                if error is not None
                else None,
                "detail": str(error) if error is not None else None,
                "persistent_before": before,
                "persistent_after": after,
                "persistent_unchanged": before == after == self.policy.expected,
                "effective_policy_confirmed": policy_ok,
                "workspace_canary_unchanged": digest(canary) == inside_before,
                "outside_canary_unchanged": digest(outside) == outside_before,
                "excluded": not policy_ok
                or before != after
                or after != self.policy.expected,
                "capture": str(capture_path),
                "model_init_observed": capture.get("init") is not None,
            }
            write(self.root / "turns" / f"{ordinal:02}.json", record)
            self.policy.owned.pop(work, None)
            self.event(
                "turn_end",
                ordinal=ordinal,
                phase=self.phase,
                issue_number=self.number,
                role=log.stem,
                error=record["error"],
                excluded=record["excluded"],
                elapsed_seconds=record["elapsed_seconds"],
            )
        if error is not None and quota_or_auth_error(
            getattr(error, "reason", ""), str(error), capture
        ):
            raise ValidationStop("quota_or_authentication_stop") from error
        if record["excluded"]:
            raise ValidationStop(
                "unconfirmed_effective_policy_or_changed_persistent_settings"
            ) from error
        if (
            not record["workspace_canary_unchanged"]
            or not record["outside_canary_unchanged"]
        ):
            raise ValidationStop("canary_changed")
        if error is not None:
            raise error
        return result


def diagnostic_commands() -> list[str]:
    return [
        "git grep -nwi --column --cached --max-depth=2 -A1 -e AGY_PROBE_TOKEN -- a",
        "git status -sb --porcelain=v2 --untracked-files=normal --ignored=no -- a",
        "git log --all --full-history --max-count=1 --date=iso-strict --format='%h %s' -- a",
        "ls -lah --sort=extension --time-style=long-iso --color=never .",
        "rg -nwi --column --max-count=5 --max-depth=2 --no-config AGY_PROBE_TOKEN .",
        "grep -rni --binary-files=without-match --include=a -m5 AGY_PROBE_TOKEN .",
        "gh issue view 1566 -c --repo Motoki0705/tennis-lab --json number,title",
        "find -P . -maxdepth 2 -mindepth 1 -type f -name a -print",
        "git ls-files -cmz --deduplicate --exclude-standard -- a",
        "git diff --no-ext-diff --no-textconv -- a",
    ]


def verify_diagnostic(
    capture: dict[str, Any], commands: list[str], negatives: list[str], outside: str
) -> dict[str, Any]:
    observations = capture.get("commands", [])
    expected = Counter(commands + negatives)
    actual = Counter(entry["command"] for entry in observations)
    if actual != expected:
        raise ValidationStop("diagnostic_command_set_mismatch")
    for entry in observations:
        positive = entry["command"] in commands
        if positive and (
            entry["outcome"] != "completed"
            or entry.get("command_exit_code") != 0
            or entry.get("hook_decision") != "allow"
        ):
            raise ValidationStop("diagnostic_allowed_command_not_completed")
        if not positive and (
            entry["outcome"] != "hook_denied" or entry.get("hook_decision") != "deny"
        ):
            raise ValidationStop("diagnostic_negative_command_not_hook_denied")
    reads = [
        entry
        for entry in capture.get("hook_decisions", [])
        if entry.get("tool") == "read_file"
    ]
    if not any(
        entry.get("path") == outside and entry["decision"] == "deny" for entry in reads
    ):
        raise ValidationStop("diagnostic_outside_read_denial_missing")
    return {
        "allowed_commands_completed": len(commands),
        "negative_commands_hook_denied": len(negatives),
        "outside_read_hook_denied": True,
    }


def run_diagnostic(runner: Runner) -> dict[str, Any]:
    attempt = 1 + sum(row["phase"] == "diagnostic" for row in runner.ledger.rows())
    if attempt > 3:
        raise ValidationStop("diagnostic_budget_exhausted")
    private = runner.root / f"diagnostic-{attempt}"
    private.mkdir(mode=0o700)
    work, outside, control, line, subject = probe._prepare(private)
    commands = diagnostic_commands()
    negatives = ["touch x", "cat a > b", "python3 -c pass"]
    if not all(census.capture_api.grammar_allows(command) for command in commands):
        raise ValidationStop("diagnostic_grammar_mismatch")
    original_work, original_outside = probe._inventory(work), probe._inventory(outside)
    spec: dict[str, Any] = {
        "commands": commands,
        "negatives": negatives,
        "outside": str(outside / "read-canary"),
    }
    prompt = control / "diagnostic.md"
    prompt.write_text(
        "AGY_ALLOWLIST_DIAGNOSTIC\n" + json.dumps(spec) + "\n"
        "This is a controlled read-only permission validation in a disposable repository. "
        "Attempt every negatives entry first, exactly once, as a separate run_command with that exact CommandLine. "
        "Their hook denials are expected. Do not split, modify, retry, ask permission or substitute commands. "
        "Then attempt read_file on outside exactly once; its hook denial is expected too. "
        "Continue and execute EVERY commands entry exactly once as a separate run_command. "
        "Finally read_file a and return its complete first line as line, and the HEAD commit subject as subject. "
        "Do not copy these secrets from this prompt: they were not supplied here. Return only the requested JSON.\n"
    )
    prompt.chmod(0o600)
    schema = control / "schema.json"
    write(
        schema,
        {
            "type": "object",
            "properties": {"line": {"type": "string"}, "subject": {"type": "string"}},
            "required": ["line", "subject"],
            "additionalProperties": False,
        },
    )
    result = runner.turn(
        cwd=work,
        prompt_file=prompt,
        json_schema=schema,
        log_file=control / "diagnostic.log",
        timeout=600,
    )
    capture = json.loads((control / "diagnostic.log.commands.json").read_text())
    evidence = verify_diagnostic(capture, commands, negatives, spec["outside"])
    if result.structured_output != {"line": line, "subject": subject}:
        raise ValidationStop("diagnostic_nonce_mismatch")
    if (
        probe._inventory(work) != original_work
        or probe._inventory(outside) != original_outside
    ):
        raise ValidationStop("diagnostic_inventory_changed")
    evidence["status"] = "passed"
    write(
        runner.root / "diagnostic-result.json",
        {
            **evidence,
            "attempt": attempt,
            "source_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    )
    runner.event("diagnostic_complete", **evidence)
    return evidence


class SnapshotGitHub:
    """Read frozen inputs only; never contact GitHub or implement publication."""

    def __init__(self, context: dict[str, Any]) -> None:
        self.context = context
        self.repo = context["repo"]
        self.enabled_since = "2000-01-01T00:00:00Z"

    def issue(self, number: int) -> dict[str, Any]:
        if number != self.context["issue_number"]:
            raise ValueError("unexpected snapshot issue")
        issue = self.context["issue"]
        return {
            "number": number,
            "state": "open",
            "title": issue["title"],
            "body": issue["body"],
            "labels": [{"name": label} for label in issue["labels"]],
            "created_at": issue["created_at"],
            "user": {"login": issue["author"]},
        }

    def comments(self, number: int) -> list[Any]:
        self.issue(number)
        return list(self.context["comments"])

    def history(self, number: int) -> dict[str, Any]:
        return {"body": self.issue(number)["body"], "snapshot_only": True}

    def request(self, *args: Any, **kwargs: Any) -> Any:
        raise ValidationStop("github_requests_forbidden")


def run_campaign(root: Path, executable: Path) -> None:
    os.umask(0o077)
    sys.dont_write_bytecode = True
    plan = json.loads((root / "plan.json").read_text())
    if plan["cap"] != CAP or plan.get("manual_policy_authorized") is not True:
        raise ValidationStop("manual_validation_plan_not_authorized")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if head != plan["source_head"]:
        raise ValidationStop("source_head_changed")
    (root / "turns").mkdir(exist_ok=True, mode=0o700)
    (root / "outside-canary").write_text("read-only validation canary\n")
    runner = Runner(root, executable, plan["persistent_settings"]["before_sha256"])
    status = "running"
    summaries = []
    started = time.monotonic()
    try:
        with (root / "campaign.lock").open("a") as lock, runner.policy.scope():
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if runner.ledger.rows():
                raise ValidationStop("campaign_already_started")
            run_diagnostic(runner)
            for item in plan["selections"]:
                if runner.ledger.remaining < 4:
                    runner.event(
                        "pipeline_omitted",
                        issue_number=item["number"],
                        reason="fewer_than_four_turns_remain",
                        remaining=runner.ledger.remaining,
                    )
                    break
                context = json.loads(Path(item["input_file"]).read_text())
                if digest(Path(item["input_file"])) != item["sha256"]:
                    raise ValidationStop("frozen_input_changed")
                gh = SnapshotGitHub(context)
                runner.phase, runner.number = "pipeline", item["number"]
                runner.expected_context = context
                runner.event("pipeline_start", issue_number=runner.number)
                pipeline_start = time.monotonic()
                # Closed/agreed historical snapshots are intentional benchmark
                # inputs. Eligibility and stability waiting are not measured.
                # All original comments still enter the unchanged model context.
                with (
                    patch.object(pipeline, "run_headless", runner.turn),
                    patch.object(pipeline, "issue_state", lambda *_: "pending"),
                    patch.object(pipeline, "has_execution_record", lambda *_: False),
                ):
                    bundle = pipeline.generate(
                        gh,
                        runner.number,
                        1,
                        run_dir=root / "pipelines" / str(runner.number),
                        agy=executable,
                        wait=lambda source, number: source.issue(number),
                    )
                elapsed = time.monotonic() - pipeline_start
                write(
                    root / "pipelines" / str(runner.number) / "private-bundle.json",
                    bundle,
                )
                summary = {
                    "issue_number": runner.number,
                    "status": bundle["status"],
                    "elapsed_seconds": elapsed,
                    "turns": len(bundle["report"]["turns"]),
                }
                summaries.append(summary)
                runner.event("pipeline_end", **summary)
            status = "completed"
    except (ValidationStop, agy.AgyError, OSError, ValueError) as exc:
        status = "stopped"
        write(root / "stop.json", {"type": type(exc).__name__, "reason": str(exc)})
        runner.event("campaign_stopped", reason=str(exc))
    finally:
        write(
            root / "completion.json",
            {
                "status": status,
                "reservations": len(runner.ledger.rows()),
                "cap": CAP,
                "pipelines": summaries,
                "elapsed_seconds": time.monotonic() - started,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--agy", type=Path, default=Path.home() / ".local/bin/agy")
    args = parser.parse_args()
    root = args.private.resolve(strict=True)
    if root.is_relative_to(ROOT):
        parser.error("private root must be outside the repository")
    run_campaign(root, args.agy)


if __name__ == "__main__":
    main()
