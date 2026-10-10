"""Command line: ``report`` (weekly, runs the agent) and ``triage`` (daily, gh only)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import agent, collect, report, shell, triage
from .shell import CommandError

JST = ZoneInfo("Asia/Tokyo")
OPS_DIR = Path(__file__).resolve().parents[2]  # .agents/ops
REPO_ROOT = OPS_DIR.parents[1]
PROMPT_FILE = OPS_DIR / "weekly-report" / "prompt.md"
LOG = logging.getLogger("weekly_report")


class ConfigError(ValueError):
    """Invalid CLI/env configuration."""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if not raw.isdigit() or int(raw) <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {raw!r}")
    return int(raw)


def _default_log_dir() -> Path:
    if os.environ.get("WEEKLY_REPORT_LOG_DIR"):
        return Path(os.environ["WEEKLY_REPORT_LOG_DIR"])
    state = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    return state / "tennis-lab-agents" / "weekly-report"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weekly_report.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("report", "triage"):
        p = sub.add_parser(name)
        p.add_argument(
            "--dry-run", action="store_true", help="do not create/edit issues"
        )
        p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
        p.add_argument("--log-dir", type=Path, default=None)
    rep = sub.choices["report"]
    rep.add_argument(
        "--agent",
        choices=agent.AGENT_CHOICES,
        default=None,
        help="default: $WEEKLY_REPORT_AGENT or 'alternate'",
    )
    rep.add_argument("--model", default=None, help="default: $WEEKLY_REPORT_MODEL")
    rep.add_argument(
        "--base-ref",
        default=None,
        help="default: $WEEKLY_REPORT_BASE_REF or origin/main",
    )
    rep.add_argument(
        "--output", type=Path, default=None, help="also write the body here"
    )
    rep.add_argument("--now", default=None, help="ISO datetime override (testing)")
    return parser


def _setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(run_dir / "run.log"),
    ):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def _run_report(args: argparse.Namespace, run_dir: Path) -> int:
    now = (
        datetime.fromisoformat(args.now).astimezone(JST)
        if args.now
        else datetime.now(JST)
    )
    setting = args.agent or os.environ.get("WEEKLY_REPORT_AGENT") or "alternate"
    agent_name = agent.resolve_agent(setting, now.date())
    cleanup_env = os.environ.get("WEEKLY_REPORT_CLEANUP_CMD")
    cfg = collect.CollectConfig(
        repo_root=args.repo_root.resolve(),
        now=now,
        base_ref=args.base_ref
        or os.environ.get("WEEKLY_REPORT_BASE_REF")
        or "origin/main",
        stale_pr_days=_env_int("WEEKLY_REPORT_STALE_PR_DAYS", 7),
        cleanup_cmd=tuple(shlex.split(cleanup_env)) if cleanup_env else None,
    )
    LOG.info(
        "report: agent=%s (setting=%s) base=%s dry_run=%s",
        agent_name,
        setting,
        cfg.base_ref,
        args.dry_run,
    )

    collected = collect.collect_all(cfg)
    (run_dir / "collected.json").write_text(
        json.dumps(collected, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    title = report.report_title(now.date())
    duplicates = [
        r for r in collected["previous_reports"]["reports"] if r["title"] == title
    ]
    if duplicates:
        msg = f"report {title!r} already exists as #{duplicates[0]['number']}"
        if not args.dry_run:
            raise ConfigError(
                msg + "; IDs would collide. Close/rename it or use --dry-run"
            )
        LOG.warning("%s (dry-run continues; a real run would fail)", msg)

    stale = report.stale_prs(collected)
    prompt = agent.build_prompt(
        PROMPT_FILE.read_text(encoding="utf-8"),
        week=title,
        stale_prs=stale,
        collected=collected,
    )
    prompt_file = run_dir / "prompt.md"
    prompt_file.write_text(prompt, encoding="utf-8")
    exec_path = Path(
        os.environ.get("WEEKLY_REPORT_AGENT_EXEC") or OPS_DIR / "lib" / "agent_exec.sh"
    )
    output = agent.execute(
        agent.AgentRun(
            agent=agent_name,
            agent_exec=exec_path,
            repo_root=cfg.repo_root,
            prompt_file=prompt_file,
            out_file=run_dir / "agent_output.md",
            log_dir=run_dir,
            model=args.model or os.environ.get("WEEKLY_REPORT_MODEL") or None,
            timeout=_env_int("WEEKLY_REPORT_AGENT_TIMEOUT", 5400),
        )
    )
    analysis = agent.parse_agent_output(output, {int(pr["number"]) for pr in stale})
    title, body, ids = report.render_report(
        now=now, agent_name=agent_name, collected=collected, analysis=analysis
    )
    (run_dir / "report.md").write_text(body, encoding="utf-8")
    LOG.info(
        "rendered %s with %d proposals (%s)", title, len(ids), run_dir / "report.md"
    )
    if args.output:
        args.output.write_text(body, encoding="utf-8")
    if args.dry_run:
        sys.stdout.write(f"# title: {title}\n\n{body}")
        return 0
    triage.ensure_labels([collect.REPORT_LABEL], repo_root=cfg.repo_root, dry_run=False)
    url = shell.gh(
        ["issue", "create", "--title", title, "--body-file", str(run_dir / "report.md"),
         "--label", collect.REPORT_LABEL],
        cwd=cfg.repo_root,
    ).strip()  # fmt: skip
    LOG.info("created report issue %s", url)
    (run_dir / "result.json").write_text(
        json.dumps({"title": title, "url": url, "ids": ids}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(url)
    return 0


def _run_triage(args: argparse.Namespace, run_dir: Path) -> int:
    result = triage.triage(
        repo_root=args.repo_root.resolve(), work_dir=run_dir, dry_run=args.dry_run
    )
    summary = {
        "created": result.created,
        "linked_existing": result.linked_existing,
        "errors": result.errors,
    }
    (run_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if result.errors else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    log_dir = args.log_dir or _default_log_dir()
    stamp = datetime.now(JST).strftime("%Y%m%dT%H%M%S")
    run_dir = (
        log_dir
        / f"{stamp}-{args.command}{'-dry' if args.dry_run else ''}-{os.getpid()}"
    )
    _setup_logging(run_dir)
    LOG.info("run dir: %s", run_dir)
    try:
        if args.command == "report":
            return _run_report(args, run_dir)
        return _run_triage(args, run_dir)
    except (CommandError, ConfigError, ValueError, OSError) as exc:
        LOG.error("%s failed: %s", args.command, exc)
        LOG.error("logs: %s", run_dir)
        return 1
