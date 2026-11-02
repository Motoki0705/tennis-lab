"""Manual, private agy census. No GitHub writes and no permission expansion."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".agents/ops/lib"))
capture_api: Any = importlib.import_module("agy_census_capture")
events: Any = importlib.import_module("agent_events")
agy_api: Any = importlib.import_module("agy_headless")
opsproc: Any = importlib.import_module("opsproc")
jsonschema: Any = importlib.import_module("jsonschema")
PHASE_LIMITS = {"pilot": 25, "measurement": 60}


def private_root() -> Path:
    return Path.home() / ".local/state/tennis-lab-agents/manual-runs/1566/agy-census"


def within_private(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(private_root().resolve()):
        raise ValueError(
            "all census artifacts and workspaces must be in the private root"
        )
    return resolved


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        path.chmod(0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def preflight(work: Path, owned_hook: dict[Path, bytes] | None = None) -> str:
    """Accept only restrictive persistent policy plus an explicit session plan.

    Unlike the production adapter, this manual runner accepts a saved auto
    startup preference. The CLI documents --mode as a session override, and we
    always send both --mode plan and agentMode=plan. It never inherits auto.
    No persistent command grants, hooks or trusted ancestors are accepted.
    """
    path = Path.home() / ".gemini/antigravity-cli/settings.json"
    raw = path.read_bytes() if path.exists() else b"{}"
    settings = events.parse_json(raw.decode("utf-8"))
    known = {
        "agentMode",
        "model",
        "trustedWorkspaces",
        "toolPermission",
        "permissions",
        "enableTerminalSandbox",
        "allowNonWorkspaceAccess",
        "$schema",
    }
    if not isinstance(settings, dict) or settings.keys() - known:
        raise ValueError("unknown persistent settings; refusing to infer their safety")
    if settings.get("agentMode", "") not in {"", "default", "plan", "auto"}:
        raise ValueError("unsupported saved startup mode")
    if settings.get("toolPermission", "request-review") not in {
        "request-review",
        "strict",
    }:
        raise ValueError("persistent toolPermission can widen permissions")
    for key in ("enableTerminalSandbox", "allowNonWorkspaceAccess"):
        if settings.get(key, False) is not False:
            raise ValueError("persistent settings differ from the read-only policy")
    permissions = settings.get("permissions", {})
    if not isinstance(permissions, dict) or any(permissions.values()):
        raise ValueError("manual census requires no persistent permission rules")
    trusted = settings.get("trustedWorkspaces", [])
    if not isinstance(trusted, list) or any(not isinstance(p, str) for p in trusted):
        raise ValueError("invalid trustedWorkspaces")
    for value in trusted:
        path = work / Path(value).expanduser()
        if work.is_relative_to(Path(os.path.abspath(path))) or work.is_relative_to(
            path.resolve()
        ):
            raise ValueError("census workspace has a trusted ancestor")
    if (Path.home() / ".gemini/config/hooks.json").exists():
        raise ValueError("global hooks would confound this census")
    for name in ("hooks.json", "mcp_config.json", "plugins"):
        extension = work / ".agents" / name
        if extension.exists() and extension not in (owned_hook or {}):
            raise ValueError("workspace extensions would confound this census")
    for path, content in (owned_hook or {}).items():
        if path.is_symlink() or path.read_bytes() != content:
            raise ValueError("census hook was modified")
    return hashlib.sha256(raw).hexdigest()


def install_deny_hook(work: Path, control: Path) -> dict[Path, bytes]:
    """Experimental arm: turn unlisted shell forms into recoverable hard errors.

    This never approves an additional form. Its changed feedback means these
    observations must be analyzed separately from the production baseline.
    """
    script = work / ".agents/agy_census_deny.py"
    settings = control / "settings.json"
    hook_log = control / "hooks.jsonl"
    code = (
        "import json,re,sys\nfrom pathlib import Path\n"
        "payload=json.load(sys.stdin)\n"
        "command=payload['toolCall']['args']['CommandLine']\n"
        f"rules=json.loads(Path({str(settings)!r}).read_text())['permissions']\n"
        "def matches(rule):\n"
        " return rule.startswith('command(regex:') and re.search(rule[14:-1],command) is not None\n"
        "allowed=not any(matches(r) for r in rules['deny']) and any(matches(r) for r in rules['allow'])\n"
        "decision='allow' if allowed else 'deny'\n"
        f"with Path({str(hook_log)!r}).open('a') as handle:\n"
        " handle.write(json.dumps({'step':payload['stepIdx'],'command':command,'decision':decision})+'\\n')\n"
        "print(json.dumps({'decision':decision,'reason':'Use permitted read-only tools; this shell form is unavailable.'}))\n"
    )
    hook_file = work / ".agents/hooks.json"
    command = shlex.join([sys.executable, str(script)])
    config = {
        "census-deny-unlisted": {
            "PreToolUse": [
                {
                    "matcher": "run_command",
                    "hooks": [{"type": "command", "command": command}],
                }
            ]
        }
    }
    owned = {script: code.encode(), hook_file: json.dumps(config).encode()}
    for path, content in owned.items():
        with path.open("xb") as handle:
            path.chmod(0o600)
            handle.write(content)
    return owned


def run_turn(
    *,
    run_id: str,
    source: Path,
    ref: str,
    agy: Path,
    inputs: Path,
    role: str,
    timeout: float,
    deny_unlisted: bool = False,
    phase: str = "pilot",
    prompt_stdin: bool = False,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run id must be one plain directory name")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if phase not in PHASE_LIMITS:
        raise ValueError("unknown census phase")
    if phase == "measurement" and (not deny_unlisted or timeout != 600):
        raise ValueError("measurement requires deny-unlisted and a 600 second limit")
    root = private_root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    with (root / "census.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phase_root = root if phase == "pilot" else root / phase
        phase_root.mkdir(exist_ok=True, mode=0o700)
        ledger = phase_root / "invocations.jsonl"
        limit = PHASE_LIMITS[phase]
        if ledger.exists() and len(ledger.read_text().splitlines()) >= limit:
            raise ValueError(f"{phase} has reached its {limit} invocation cap")
        directory = within_private(phase_root / "runs" / run_id)
        directory.mkdir(parents=True, mode=0o700)
        work, control = directory / "work", directory / "control"
        control.mkdir(mode=0o700)
        git_env = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }

        def git(*args: str, cwd: Path = source) -> str:
            return subprocess.check_output(
                ["git", "-c", "core.hooksPath=/dev/null", *args],
                cwd=cwd,
                env=git_env,
                text=True,
                stderr=subprocess.PIPE,
                timeout=60,
            )

        commit = git("rev-parse", "--verify", ref + "^{commit}").strip()
        git(
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(source.resolve()),
            str(work),
        )
        git("checkout", "--quiet", "--detach", commit, cwd=work)
        git(
            "remote",
            "set-url",
            "origin",
            "https://github.com/Motoki0705/tennis-lab",
            cwd=work,
        )
        before_settings = preflight(work)
        persistent = Path.home() / ".gemini/antigravity-cli/settings.json"
        saved_bytes = persistent.read_bytes() if persistent.exists() else b"{}"
        if hashlib.sha256(saved_bytes).hexdigest() != before_settings:
            raise ValueError("persistent settings changed during preparation")
        with (control / "saved-settings-before.json").open("xb") as saved_handle:
            os.fchmod(saved_handle.fileno(), 0o600)
            saved_handle.write(saved_bytes)
        context = events.parse_json(within_private(inputs).read_text())
        assets = ".agents/ops/issue_rewrite/"
        template = git("show", commit + ":" + assets + f"prompts/{role}.md")
        schema = git("show", commit + ":" + assets + f"schemas/{role}.schema.json")
        prompt = (
            template + "\n\n入力（JSON）:\n" + json.dumps(context, ensure_ascii=False)
        )
        prompt_file, schema_file = control / "prompt.md", control / "schema.json"
        prompt_file.write_text(prompt, encoding="utf-8")
        schema_file.write_text(schema, encoding="utf-8")
        for path in (prompt_file, schema_file):
            path.chmod(0o600)
        settings = agy_api.readonly_settings()
        write_json(control / "settings.json", settings)
        owned_hook = install_deny_hook(work, control) if deny_unlisted else None
        config = agy_api.load_config()
        env = {
            **os.environ,
            "AGY_CLI_DISABLE_AUTO_UPDATE": "true",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GH_PAGER": "cat",
            "RIPGREP_CONFIG_PATH": "/dev/null",
        }
        for key in list(env):
            if key.startswith(("HERDR_", "CLAUDE", "CODEX")) or key in {
                "GH_TOKEN",
                "GITHUB_TOKEN",
            }:
                env.pop(key)

        def version() -> str:
            value = subprocess.check_output(
                [str(agy), "--version"],
                cwd=work,
                env=env,
                text=True,
                timeout=15,
            ).strip()
            if value != config.version:
                raise ValueError("agy version differs from the production pin")
            return value

        version()
        log_file = control / "agy.log"
        command = [
            str(agy.resolve()),
            *(["--input-format", "stream-json"] if prompt_stdin else ["-p", prompt]),
            "--model",
            config.model,
            "--mode",
            "plan",
            "--settings",
            json.dumps(settings),
            "--output-format",
            "stream-json",
            "--json-schema",
            str(schema_file),
            "--log-file",
            str(log_file),
        ]
        stdin_file = control / "stdin.jsonl"
        if prompt_stdin:
            with stdin_file.open("x", encoding="utf-8") as handle:
                stdin_file.chmod(0o600)
                handle.write(
                    json.dumps(
                        {"event": "user", "message": {"content": prompt}},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        write_json(
            control / "metadata.json",
            {
                "run_id": run_id,
                "role": role,
                "source_commit": commit,
                "issue_number": context["issue_number"],
                "version": config.version,
                "model": config.model,
                "arm": "deny_unlisted" if deny_unlisted else "production",
                "phase": phase,
                "timeout_seconds": timeout,
                "prompt_transport": "stdin_stream" if prompt_stdin else "argv",
                "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
                "saved_settings_sha256": before_settings,
            },
        )
        opsproc.enable_child_subreaper()
        started = time.monotonic()
        timed_out = False
        with ledger.open("a", encoding="utf-8") as handle:
            ledger.chmod(0o600)
            handle.write(json.dumps({"run": "runs/" + run_id, "role": role}) + "\n")
            handle.flush()
        stdout_path, stderr_path = control / "stdout.jsonl", control / "stderr.log"
        with (
            stdout_path.open("x") as stdout,
            stderr_path.open("x") as stderr,
            (stdin_file if prompt_stdin else Path(os.devnull)).open("rb") as stdin,
        ):
            stdout_path.chmod(0o600)
            stderr_path.chmod(0o600)
            proc = subprocess.Popen(
                command,
                cwd=work,
                env=env,
                stdout=stdout,
                stderr=stderr,
                stdin=stdin,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                opsproc._terminate_group(proc, proc.pid, 5)
                opsproc.terminate_descendants(5)
        elapsed = time.monotonic() - started
        version()
        settings_unchanged = preflight(work, owned_hook) == before_settings
        log = log_file.read_text()
        log_file.chmod(0o600)
        observed_model = agy_api._observed_model(log_file, config.model, [])
        capture: dict[str, Any] = capture_api.capture_stream(
            stdout_path.read_text(), log
        )
        init = capture["init"] or {}
        if (
            init.get("permission_mode") != "request-review"
            or "applying agent mode plan" not in log
        ):
            raise ValueError(
                "log and init do not confirm the requested permission mode"
            )
        result = capture["result"] or {}
        schema_error = None
        try:
            jsonschema.validate(result.get("structured_output"), json.loads(schema))
        except jsonschema.ValidationError as error:
            schema_error = error.message
        capture.update(
            run_id=run_id,
            role=role,
            issue_number=context["issue_number"],
            exit_code=proc.returncode,
            timed_out=timed_out,
            elapsed_seconds=elapsed,
            within_300_seconds=elapsed <= 300,
            timeout_seconds=timeout,
            phase=phase,
            prompt_transport="stdin_stream" if prompt_stdin else "argv",
            persistent_settings_unchanged=settings_unchanged,
            observed_model=observed_model,
            arm="deny_unlisted" if deny_unlisted else "production",
            schema_error=schema_error,
            structured_valid=schema_error is None,
            checkout_status=git(
                "status", "--porcelain", "--untracked-files=all", cwd=work
            ),
        )
        if (control / "hooks.jsonl").exists():
            (control / "hooks.jsonl").chmod(0o600)
        write_json(control / "capture.json", capture)
        if not settings_unchanged:
            raise ValueError("persistent agy settings changed during the run")
        return capture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run")
    run.add_argument("--id", required=True)
    run.add_argument("--source", type=Path, default=ROOT)
    run.add_argument("--ref", default="HEAD")
    run.add_argument("--agy", type=Path, required=True)
    run.add_argument("--input-json", type=Path, required=True)
    run.add_argument("--role", choices=("writer", "verifier"), required=True)
    run.add_argument("--timeout", type=float, default=300)
    run.add_argument("--phase", choices=tuple(PHASE_LIMITS), default="pilot")
    run.add_argument(
        "--prompt-stdin",
        action="store_true",
        help="send exactly one unchanged prompt on stdin",
    )
    run.add_argument(
        "--deny-unlisted",
        action="store_true",
        help="experimental denial feedback; analyze separately",
    )
    summary = actions.add_parser("summarize")
    summary.add_argument("--captures", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        capture = run_turn(
            run_id=args.id,
            source=args.source,
            ref=args.ref,
            agy=args.agy,
            inputs=args.input_json,
            role=args.role,
            timeout=args.timeout,
            deny_unlisted=args.deny_unlisted,
            phase=args.phase,
            prompt_stdin=args.prompt_stdin,
        )
        print(
            json.dumps(
                {
                    key: capture[key]
                    for key in (
                        "run_id",
                        "exit_code",
                        "timed_out",
                        "structured_valid",
                        "elapsed_seconds",
                    )
                }
            )
        )
    else:
        captures = [
            json.loads(within_private(path).read_text()) for path in args.captures
        ]
        write_json(within_private(args.output), capture_api.summarize(captures))


if __name__ == "__main__":
    main()
