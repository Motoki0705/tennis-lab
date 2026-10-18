"""Command line: ``report`` (three-day slots), ``schedule`` (local state) and ``triage`` (daily, gh only)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any

from . import agent, collect, lock, proposals, report, schedule, shell, triage
from .shell import CommandError

JST = schedule.JST
OPS_DIR = Path(__file__).resolve().parents[2]  # .agents/ops
REPO_ROOT = OPS_DIR.parents[1]
PROMPT_FILE = OPS_DIR / "ops-report" / "prompt.md"
LOG = logging.getLogger("ops_report")


class ConfigError(ValueError):
    """Invalid CLI/env configuration."""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if not raw.isdigit() or int(raw) <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {raw!r}")
    return int(raw)


def _state_dir() -> Path:
    return schedule.state_dir().parent


def _default_log_dir() -> Path:
    if os.environ.get("OPS_REPORT_LOG_DIR"):
        return Path(os.environ["OPS_REPORT_LOG_DIR"])
    return _state_dir() / "ops-report"


def _issue_lock() -> AbstractContextManager[None]:
    """Lock shared by every run on this machine (not per log dir / checkout)."""
    path = Path(
        os.environ.get("OPS_REPORT_LOCK_FILE") or _state_dir() / "ops-report.lock"
    )
    return lock.exclusive(path, timeout=_env_int("OPS_REPORT_LOCK_TIMEOUT", 600))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ops_report.py", description=__doc__)
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
        help="default: $OPS_REPORT_AGENT or 'codex'",
    )
    rep.add_argument(
        "--base-ref",
        default=None,
        help="default: $OPS_REPORT_BASE_REF or origin/main",
    )
    rep.add_argument(
        "--output", type=Path, default=None, help="also write the body here"
    )
    rep.add_argument(
        "--now", default=None, help="timezone-aware ISO datetime (testing)"
    )
    rep.add_argument(
        "--if-due",
        action="store_true",
        help="no-op before the first slot or if already published",
    )
    sched = sub.add_parser("schedule")
    actions = sched.add_subparsers(dest="action", required=True)
    for name in ("init", "status"):
        action = actions.add_parser(name)
        action.add_argument(
            "--now", default=None, help="timezone-aware ISO datetime (testing)"
        )
        action.add_argument("--repo-root", type=Path, default=REPO_ROOT)
        action.add_argument("--log-dir", type=Path, default=None)
        action.set_defaults(dry_run=False)
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


def _now(args: argparse.Namespace) -> datetime:
    return schedule.aware_time(args.now) if args.now else datetime.now(JST)


def _noop(run_dir: Path, status: str, slot: datetime | None) -> int:
    result = {"status": status, "slot": slot.isoformat() if slot else None}
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    LOG.info("report: %s (%s)", status, slot)
    print(json.dumps(result))
    return 0


def _run_report(args: argparse.Namespace, run_dir: Path) -> int:
    path = schedule.state_dir() / "schedule.json"
    # Serialize collection as well as publication: even a run crossing into the
    # next slot must not let two reports collect from the same previous boundary.
    with lock.exclusive(
        path.with_suffix(".lock"), timeout=_env_int("OPS_REPORT_LOCK_TIMEOUT", 600)
    ):
        state = schedule.Schedule.load(path)
        now = _now(args).replace(microsecond=0)
        slot = schedule.latest_slot(state.anchor, now)
        if slot is None:
            if args.if_due:
                return _noop(run_dir, "not_due", None)
            raise ConfigError(
                f"first slot is {state.anchor.isoformat()}; report is not due"
            )
        previous = state.data["last_attempt"]
        if previous and schedule.aware_time(previous["slot"]) > slot:
            raise ConfigError("clock moved behind the last attempted slot")
        success = state.data["last_success"]
        attempted = not args.dry_run and (
            not success or success["slot"] != slot.isoformat()
        )
        if attempted:
            state.begin(slot, now)
        try:
            history = collect.collect_report_history(args.repo_root.resolve())
            if success and not any(
                r["slot"]
                == proposals.slot_tag(schedule.aware_time(success["slot"]).date())
                for r in history["reports"]
            ):
                raise ConfigError(
                    "local success has no published report; resolve the missing GitHub issue explicitly"
                )
            if not args.dry_run:
                state.reconcile(
                    slot,
                    now,
                    {
                        schedule.aware_time(f"{r['slot']}T08:00:00+09:00"): r["url"]
                        for r in history["reports"]
                    },
                )
            tag = proposals.slot_tag(slot.date())
            published = next((r for r in history["reports"] if r["slot"] == tag), None)
            if published:
                msg = f"report for {tag} already exists as #{published['number']}"
                if attempted:
                    # Recover a crash after gh create but before local commit.
                    state.finish(_now(args), url=published["url"])
                    attempted = False
                if args.if_due:
                    return _noop(run_dir, "already_published", slot)
                if not args.dry_run:
                    raise ConfigError(msg + "; use --dry-run to preview")
                LOG.warning("%s (preview only)", msg)
            url = _generate_report(
                args,
                run_dir,
                now,
                slot,
                (slot - state.anchor) // schedule.PERIOD,
                history,
                state.anchor - schedule.PERIOD,
            )
            if attempted:
                state.finish(_now(args), url=url)
                attempted = False
        except BaseException as exc:
            if attempted:
                state.finish(_now(args), error=f"{type(exc).__name__}: {exc}")
            raise
        return 0


def _run_schedule(args: argparse.Namespace) -> int:
    path = schedule.state_dir() / "schedule.json"
    now = _now(args)
    if args.action == "init":
        with lock.exclusive(
            path.with_suffix(".lock"), timeout=_env_int("OPS_REPORT_LOCK_TIMEOUT", 600)
        ):
            state = schedule.Schedule.initialize(path, now)
    else:
        state = schedule.Schedule.load(path)
    print(json.dumps(state.status(now), ensure_ascii=False, indent=2))
    return 0


def _generate_report(
    args: argparse.Namespace,
    run_dir: Path,
    now: datetime,
    slot: datetime,
    slot_index: int,
    history: dict[str, Any],
    initial_since: datetime,
) -> str | None:
    setting = args.agent or os.environ.get("OPS_REPORT_AGENT") or "codex"
    agent_name = agent.resolve_agent(setting, slot_index)
    cleanup_env = os.environ.get("OPS_REPORT_CLEANUP_CMD")
    repo_root = args.repo_root.resolve()
    base = collect.pin_base(
        repo_root,
        args.base_ref or os.environ.get("OPS_REPORT_BASE_REF") or "origin/main",
    )
    cfg = collect.CollectConfig(
        repo_root=repo_root,
        now=now,
        base=base,
        stale_pr_days=_env_int("OPS_REPORT_STALE_PR_DAYS", 7),
        cleanup_cmd=tuple(shlex.split(cleanup_env)) if cleanup_env else None,
    )
    LOG.info(
        "report: agent=%s (setting=%s) base=%s@%s (fetched: %s) dry_run=%s",
        agent_name,
        setting,
        base.name,
        base.sha,
        base.fetched_from or "no, local ref",
        args.dry_run,
    )

    collected = collect.collect_all(cfg, history=history, initial_since=initial_since)
    (run_dir / "collected.json").write_text(
        json.dumps(collected, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    title = report.report_title(slot.date())
    tag = proposals.slot_tag(slot.date())

    stale = report.stale_prs(collected)
    prompt = agent.build_prompt(
        PROMPT_FILE.read_text(encoding="utf-8"),
        slot=slot.date().isoformat(),
        stale_prs=stale,
        collected=collected,
    )
    prompt_file = run_dir / "prompt.md"
    prompt_file.write_text(prompt, encoding="utf-8")
    exec_path = Path(
        os.environ.get("OPS_REPORT_AGENT_EXEC") or OPS_DIR / "lib" / "agent_exec.sh"
    )
    output = agent.execute(
        agent.AgentRun(
            agent=agent_name,
            agent_exec=exec_path,
            repo_root=cfg.repo_root,
            prompt_file=prompt_file,
            out_file=run_dir / "agent_output.md",
            log_dir=run_dir,
            timeout=_env_int("OPS_REPORT_AGENT_TIMEOUT", 5400),
        )
    )
    analysis = agent.parse_agent_output(output, {int(pr["number"]) for pr in stale})
    title, body, ids = report.render_report(
        slot=slot.date(), agent_name=agent_name, collected=collected, analysis=analysis
    )
    (run_dir / "report.md").write_text(body, encoding="utf-8")
    LOG.info(
        "rendered %s with %d proposals (%s)", title, len(ids), run_dir / "report.md"
    )
    if args.output:
        args.output.write_text(body, encoding="utf-8")
    if args.dry_run:
        sys.stdout.write(f"# title: {title}\n\n{body}")
        return None
    with _issue_lock():
        fresh = collect.collect_report_history(cfg.repo_root)
        duplicate = next((r for r in fresh["reports"] if r["slot"] == tag), None)
        if duplicate:
            raise ConfigError(
                f"report for {tag} already exists as #{duplicate['number']}"
            )
        if [(r["slot"], r["period_end"]) for r in fresh["reports"]] != [
            (r["slot"], r["period_end"]) for r in history["reports"]
        ]:
            raise ConfigError(
                "report history changed during collection; retry from the latest boundary"
            )
        triage.ensure_labels(
            [collect.REPORT_LABEL], repo_root=cfg.repo_root, dry_run=False
        )
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
    return url


def _run_triage(args: argparse.Namespace, run_dir: Path) -> int:
    # Child lookup -> create -> report annotation must not interleave with another run.
    with _issue_lock():
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
    log_dir = (args.log_dir or _default_log_dir()).resolve()
    stamp = datetime.now(JST).strftime("%Y%m%dT%H%M%S")
    run_dir = (
        log_dir
        / f"{stamp}-{args.command}{'-dry' if args.dry_run else ''}-{os.getpid()}"
    )
    _setup_logging(run_dir)
    LOG.info("run dir: %s", run_dir)
    try:
        legacy = [name for name in os.environ if name.startswith("WEEKLY_REPORT_")]
        if legacy:
            raise ConfigError(
                f"unsupported legacy environment variables {legacy}; rename to OPS_REPORT_*"
            )
        if args.command == "schedule":
            return _run_schedule(args)
        if args.command == "report":
            return _run_report(args, run_dir)
        return _run_triage(args, run_dir)
    except (CommandError, ConfigError, ValueError, OSError) as exc:
        LOG.error("%s failed: %s", args.command, exc)
        LOG.error("logs: %s", run_dir)
        return 1
    except Exception:
        # Unexpected shapes (e.g. gh JSON changes) still land in run.log, then fail.
        LOG.exception(
            "%s failed with an unexpected error; logs: %s", args.command, run_dir
        )
        return 1
