"""Command line: three-day reports, local scheduling and daily adopted-task triage."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any

from . import (
    agent,
    audit,
    collect,
    lock,
    memory,
    memory_pr,
    proposals,
    report,
    schedule,
    shell,
    triage,
)
from .ledger import Ledger
from .retrieval import strict_json
from .shell import CommandError

JST = schedule.JST
OPS_DIR = Path(__file__).resolve().parents[2]  # .agents/ops
REPO_ROOT = OPS_DIR.parents[1]
PROMPT_FILE = OPS_DIR / "ops-report" / "prompt.md"
LOG = logging.getLogger("ops_report")


def _interrupt(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


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
    return Path(schedule.state_dir()).parent


def _default_log_dir() -> Path:
    if os.environ.get("OPS_REPORT_LOG_DIR"):
        return Path(os.environ["OPS_REPORT_LOG_DIR"])
    return _state_dir() / "ops-report"


def _issue_lock() -> AbstractContextManager[None]:
    """Lock shared by every run on this machine (not per log dir / checkout)."""
    path = Path(
        os.environ.get("OPS_REPORT_LOCK_FILE") or _state_dir() / "ops-report.lock"
    )
    context: AbstractContextManager[None] = lock.exclusive(
        path, timeout=_env_int("OPS_REPORT_LOCK_TIMEOUT", 600)
    )
    return context


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
    rep.add_argument(
        "--memory-decisions",
        type=Path,
        default=None,
        help="optional triage decisions snapshot; must still match GitHub",
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
        try:
            history = collect.collect_report_history(args.repo_root.resolve())
        except BaseException as exc:
            if not args.dry_run:
                previous = state.data["last_attempt"]
                failed_slot = (
                    schedule.aware_time(previous["slot"])
                    if previous and previous.get("incomplete")
                    else slot
                )
                state.begin(failed_slot, now)
                state.finish(
                    _now(args),
                    error=f"{type(exc).__name__}: {exc}",
                    incomplete=bool(previous and previous.get("incomplete")),
                    url=previous.get("url")
                    if previous and previous.get("incomplete")
                    else None,
                )
            raise
        partials = [r for r in history["reports"] if r["status"] == "incomplete"]
        if partials:
            # A published incomplete run is owed work, even across later slots.
            slot = schedule.aware_time(f"{partials[0]['slot']}T08:00:00+09:00")
            if (
                slot > now
                or slot < state.anchor
                or (slot - state.anchor) % schedule.PERIOD
            ):
                raise ConfigError("incomplete report is outside this schedule")
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
            store = memory_pr.Store(args.repo_root.resolve())
            memory_errors = [] if args.dry_run else store.sync(history, run_dir)
            if success and not any(
                r["slot"]
                == proposals.slot_tag(schedule.aware_time(success["slot"]).date())
                and r["status"] == "complete"
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
                        if r["status"] == "complete"
                    },
                )
            tag = proposals.slot_tag(slot.date())
            published = next((r for r in history["reports"] if r["slot"] == tag), None)
            if published and published["status"] == "complete":
                checkpoint = schedule.state_dir() / "audit" / f"{tag}.json"
                if not args.dry_run and checkpoint.with_suffix(".ledger.json").exists():
                    saved = audit.load_checkpoint(checkpoint, args.repo_root.resolve())
                    if (
                        saved["collected"]["audit_input_sha256"] != published["audit"]
                        or saved["collected"]["meta"]["base_sha"]
                        != published["base_sha"]
                    ):
                        raise ConfigError(
                            "published report and saved ledger input disagree"
                        )
                    book = Ledger.load(
                        Path(saved["ledger"]), slot=tag, base_sha=published["base_sha"]
                    )
                    bound = book.snapshot["run"]["report_issue"]
                    if bound is None:
                        # Recover publication followed by a crash before local binding.
                        book.bind_report(published["number"], at=audit.timestamp())
                    elif bound != published["number"]:
                        raise ConfigError(
                            "published report and ledger report issue disagree"
                        )
                msg = f"report for {tag} already exists as #{published['number']}"
                if attempted:
                    # Recover a crash after gh create but before local commit.
                    state.finish(_now(args), url=published["url"])
                    attempted = False
                if args.if_due:
                    _noop(run_dir, "already_published", slot)
                    return 1 if memory_errors else 0
                if not args.dry_run:
                    raise ConfigError(msg + "; use --dry-run to preview")
                LOG.warning("%s (preview only)", msg)
            url, save_errors, complete = _generate_report(
                args,
                run_dir,
                now,
                slot,
                history,
                state.anchor - schedule.PERIOD,
                store,
            )
            if attempted:
                state.finish(
                    _now(args),
                    url=url,
                    incomplete=not complete,
                    error=None
                    if complete
                    else f"未完了レポート {url}; report --if-due で不足分を再開",
                )
                attempted = False
        except BaseException as exc:
            if attempted:
                state.finish(
                    _now(args),
                    error=f"{type(exc).__name__}: {exc}",
                    incomplete=bool(partials),
                    url=partials[0]["url"] if partials else None,
                )
            raise
        return 1 if memory_errors or save_errors or not complete else 0


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
    history: dict[str, Any],
    initial_since: datetime,
    store: memory_pr.Store,
) -> tuple[str | None, list[str], bool]:
    setting = args.agent or os.environ.get("OPS_REPORT_AGENT") or "codex"
    agent_name = agent.resolve_agent(setting)
    repo_root = args.repo_root.resolve()
    if "OPS_REPORT_AGENT_TIMEOUT" in os.environ:
        raise ConfigError(
            "OPS_REPORT_AGENT_TIMEOUT was removed; whole-audit elapsed-time limits are not supported"
        )
    for removed in ("OPS_REPORT_CLEANUP_CMD", "OPS_REPORT_STALE_PR_DAYS"):
        if removed in os.environ:
            raise ConfigError(
                f"{removed} was removed with bulk collection; use the bounded retrieval tools"
            )
    tag = proposals.slot_tag(slot.date())
    checkpoint = schedule.state_dir() / "audit" / f"{tag}.json"
    published = next(
        (
            r
            for r in history["reports"]
            if r["slot"] == tag and r["status"] == "incomplete"
        ),
        None,
    )
    state: dict[str, Any] | None = None
    collected: dict[str, Any] | None = None
    if checkpoint.exists():
        state = audit.load_checkpoint(checkpoint, repo_root)
        collected = state["collected"]
        meta = collected["meta"]
        base = collect.BaseRef(
            meta["base_ref"], meta["base_sha"], meta["base_fetched_from"]
        )
        requested_ref = args.base_ref or os.environ.get("OPS_REPORT_BASE_REF")
        if requested_ref and requested_ref != base.name:
            raise ConfigError("cannot change the base ref of a resumable audit")
    elif published:
        raise ConfigError(
            f"incomplete report #{published['number']} has no audit checkpoint {checkpoint}; restore its version 2 ledger and acquisition records explicitly (version 1 migration: #1458)"
        )
    else:
        base = collect.pin_base(
            repo_root,
            args.base_ref or os.environ.get("OPS_REPORT_BASE_REF") or "origin/main",
        )
    if published:
        if (
            not collected
            or published["base_sha"] != base.sha
            or published["audit"] != collected["audit_input_sha256"]
        ):
            raise ConfigError("incomplete report does not match its pinned audit input")
        if published["state"] != "OPEN":
            raise ConfigError(
                "incomplete report is closed; reopen the existing issue before resuming"
            )
        _, parsed = proposals.parse_report(published["body"])
        if any(
            p.child or p.head_checked or any(checked for _, checked in p.options)
            for p in parsed
        ):
            raise ConfigError(
                "incomplete report has selections/children; resolve them explicitly before replacing proposals"
            )
    cfg = collect.CollectConfig(repo_root=repo_root, now=now, base=base)
    from agent_settings import read_pins

    pins = read_pins(OPS_DIR / "lib" / "agents.conf")
    raw_threads = pins["OPS_REPORT_MAX_THREADS"]
    if not raw_threads.isdecimal() or int(raw_threads) < 2:
        raise ConfigError("OPS_REPORT_MAX_THREADS must be >= 2")
    max_threads = int(raw_threads)
    if collected is None:
        collected = collect.collect_all(
            cfg, history=history, initial_since=initial_since
        )
        collected["ops_memory"]["persistence"] = store.describe(history)
        decisions = triage.collect_decisions(repo_root, history)
        for decision in memory.load_decisions(args.memory_decisions):
            if decision not in decisions:
                raise ConfigError("memory decisions snapshot no longer matches GitHub")
        collected["ops_memory"]["decisions"] = decisions
        collected["previous_reports"] = [
            {k: r[k] for k in ("slot", "number", "base_sha", "status", "url")}
            for r in history["reports"]
            if r["slot"] != tag
        ]
        collected["audit_input_sha256"] = audit.input_digest(collected)
    elif args.memory_decisions:
        raise ConfigError(
            "cannot replace decisions in a pinned audit; resume with the original inputs"
        )
    # A preview writes only its own ledger copy, including when resuming a real round.
    source_checkpoint = checkpoint
    if args.dry_run:
        checkpoint = run_dir / "preview" / f"{tag}.json"
    round_dir = checkpoint.parent / tag
    round_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    ledger_path = checkpoint.with_suffix(".ledger.json").absolute()
    if state is None:
        book = Ledger.create(
            ledger_path,
            slot=tag,
            base_sha=base.sha,
            goal=audit.GOAL,
            at=audit.timestamp(),
        )
        state = {
            "version": audit.VERSION,
            "collected": collected,
            "ledger": str(ledger_path),
            "attempts": [],
        }
    else:
        if args.dry_run:
            shutil.copyfile(state["ledger"], ledger_path)
            shutil.copytree(
                source_checkpoint.parent / tag, round_dir, dirs_exist_ok=True
            )
            state = {**state, "ledger": str(ledger_path)}
        book = Ledger.load(ledger_path, slot=tag, base_sha=base.sha)
    if published:
        bound = book.snapshot["run"]["report_issue"]
        if bound is None:
            book.bind_report(published["number"], at=audit.timestamp())
        elif bound != published["number"]:
            raise ConfigError("published report and ledger report issue disagree")
    memory_pr._atomic_json(checkpoint, state)
    input_path = round_dir / "input.json"
    memory_pr._atomic_json(input_path, collected)
    memory_pr._atomic_json(run_dir / "collected.json", collected)
    completion_path = round_dir / "completion.json"
    errors: list[str] = []
    if book.snapshot["run"]["status"] == "completed":
        # Publication may have failed after the ledger was completed. Reuse the
        # exact validated result, rather than starting a new audit or issue.
        update = memory.parse_update(
            strict_json(completion_path.read_bytes())["memory_update"]
        )
        errors, _ = audit.verify(
            book,
            state["attempts"],
            round_dir / "sources" / "retrieval.jsonl",
            max_threads,
        )
        if errors:
            raise ConfigError(
                "completed audit evidence changed; restore explicitly: "
                + "; ".join(errors)
            )
        complete = True
    else:
        if state["attempts"]:
            audit.ensure_quiescent(state["attempts"])
        if book.snapshot["run"]["status"] == "incomplete":
            book.resume(
                "同じ予定回・基準版・issueで不足の調査を再開", at=audit.timestamp()
            )
        if state["attempts"]:
            for directory in state["attempts"]:
                audit.capture_unreceived(book, Path(directory))
            audit.interrupt_tasks(book, "前の親runが終了したため未受領タスクを再開する")
        # Persist the attempt before launching, so even a killed parent has a
        # recovery reference. The runner remains read-only and owns its cleanup.
        runtime_dir = run_dir / "agent-runtime"
        state["attempts"].append(str(runtime_dir))
        memory_pr._atomic_json(checkpoint, state)
        ending_path = run_dir / "ending.json"
        snapshot_path = audit.worktree_path(repo_root, run_dir.name)
        with audit.checkout(repo_root, base.sha, snapshot_path) as snapshot:
            connection = {
                "command": sys.executable,
                "args": [
                    "-B",
                    "-m",
                    "opsreport.mcp_server",
                    "--repo",
                    str(snapshot),
                    "--base-sha",
                    base.sha,
                    "--repository",
                    collected["meta"]["repository"],
                    "--round-id",
                    tag,
                    "--journal-dir",
                    str(round_dir / "sources"),
                    "--ops-memory-sha",
                    collected["ops_memory"].get("base_sha", base.sha),
                    "--ledger-path",
                    str(ledger_path),
                    "--runtime-dir",
                    str(runtime_dir),
                    "--input-path",
                    str(input_path),
                    "--ending-path",
                    str(ending_path),
                    "--max-threads",
                    str(max_threads),
                ],
                "env": {"PYTHONPATH": str(PROMPT_FILE.parent)},
            }
            mcp_config = run_dir / "mcp.json"
            memory_pr._atomic_json(mcp_config, {"ops_report": connection})
            prompt = agent.build_prompt(
                PROMPT_FILE.read_text(encoding="utf-8"),
                metadata={
                    "slot": tag,
                    "base_sha": base.sha,
                    "repository": collected["meta"]["repository"],
                    "period_start": collected["meta"]["period_start"],
                    "period_end": collected["meta"]["period_end"],
                    "report_issue": book.snapshot["run"]["report_issue"],
                    "input_sha256": collected["audit_input_sha256"],
                    "output": "MCPの台帳へ記録。レポート描画・公開はホストが行う。",
                    "max_threads": max_threads,
                },
            )
            prompt_file = run_dir / "prompt.md"
            prompt_file.write_text(prompt, encoding="utf-8")
            output = ""
            try:
                output = agent.execute(
                    agent.AgentRun(
                        agent=agent_name,
                        agent_exec=Path(
                            os.environ.get("OPS_REPORT_AGENT_EXEC")
                            or OPS_DIR / "lib" / "agent_exec.sh"
                        ).resolve(),
                        repo_root=snapshot,
                        prompt_file=prompt_file,
                        out_file=run_dir / "agent_output.md",
                        log_dir=run_dir,
                        mcp_config=mcp_config,
                    )
                )
            except (CommandError, KeyboardInterrupt) as exc:
                errors.append(f"親の実行・中断: {type(exc).__name__}: {exc}")
                LOG.error("audit interrupted/failed: %s", errors[-1])
            # Read-only agents persist through MCP, even when the final reply is lost.
            book = Ledger.load(ledger_path, slot=tag, base_sha=base.sha)
            audit.capture_unreceived(book, runtime_dir)
            update = None
            try:
                update = agent.parse_completion(output)
                memory.apply_update(
                    collected["ops_memory"]["files"],
                    update,
                    source=f"ops-report:{tag}",
                    today=now.date(),
                )
            except ValueError as exc:
                errors.append(f"専用memoryの統合: {exc}")
                update = None
            verified_errors, invalid_findings = audit.verify(
                book,
                state["attempts"],
                round_dir / "sources" / "retrieval.jsonl",
                max_threads,
            )
            errors.extend(verified_errors)
            ending = (
                strict_json(ending_path.read_bytes()) if ending_path.exists() else None
            )
            memory_pr._atomic_json(completion_path, {"memory_update": update})
            complete = audit.finish(book, ending, errors, invalid_findings)
        for error in errors:
            LOG.error("audit incomplete: %s", error)
    memory_pr._atomic_json(
        run_dir / "audit.json",
        {"complete": complete, "errors": errors, "ledger": book.snapshot},
    )
    title, body, ids = report.render_ledger_report(
        ledger=book,
        agent_name=agent_name,
        since=datetime.fromisoformat(collected["meta"]["period_start"]),
        until=datetime.fromisoformat(collected["meta"]["period_end"]),
        input_sha256=collected["audit_input_sha256"],
        ledger_reference=f"予定回 {tag} の保存台帳（発見・タスクIDで照合。公開レポートの管理者が原本を保管）",
    )
    body = report.with_memory_status(body, collected["ops_memory"])
    (run_dir / "report.md").write_text(body, encoding="utf-8")
    LOG.info(
        "rendered %s with %d proposals (%s)", title, len(ids), run_dir / "report.md"
    )
    if args.output:
        args.output.write_text(body, encoding="utf-8")
    if args.dry_run:
        sys.stdout.write(f"# title: {title}\n\n{body}")
        return None, [], complete
    with _issue_lock():
        fresh = collect.collect_report_history(cfg.repo_root)
        duplicate = next((r for r in fresh["reports"] if r["slot"] == tag), None)
        if duplicate and (
            not published
            or duplicate["number"] != published["number"]
            or duplicate["status"] != "incomplete"
        ):
            raise ConfigError(
                f"report for {tag} already exists as #{duplicate['number']}"
            )
        if [(r["slot"], r["body"]) for r in fresh["reports"]] != [
            (r["slot"], r["body"]) for r in history["reports"]
        ]:
            raise ConfigError(
                "report history changed during collection; retry from the latest boundary"
            )
        triage.ensure_labels(
            [collect.REPORT_LABEL], repo_root=cfg.repo_root, dry_run=False
        )
        if complete:
            store.prepare(
                tag,
                collected["ops_memory"],
                update,
                title=title,
                body=body,
            )
            if duplicate:
                # Pin the edit target even if the edit succeeds but its response is lost.
                store.record_publication(tag, duplicate["url"])
        if duplicate:
            shell.gh(
                [
                    "issue",
                    "edit",
                    str(duplicate["number"]),
                    "--title",
                    title,
                    "--body-file",
                    str(run_dir / "report.md"),
                ],
                cwd=repo_root,
            )
            url = duplicate["url"]
        else:
            url = shell.gh(
                ["issue", "create", "--title", title, "--body-file", str(run_dir / "report.md"),
                 "--label", collect.REPORT_LABEL], cwd=cfg.repo_root,
            ).strip()  # fmt: skip
            if complete:
                store.record_publication(tag, url)
    issue_number = int(url.rstrip("/").rsplit("/", 1)[-1])
    if book.snapshot["run"]["report_issue"] is None:
        book.bind_report(issue_number, at=audit.timestamp())
    LOG.info("created report issue %s", url)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "title": title,
                "url": url,
                "ids": ids,
                "status": "complete" if complete else "incomplete",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(url)
    # The schedule records publication success even if memory persistence fails.
    # A retry recovers via history, without republishing or rerunning the agent.
    save_errors = store.sync(collect.collect_report_history(cfg.repo_root), run_dir)
    return url, save_errors, complete


def _run_triage(args: argparse.Namespace, run_dir: Path) -> int:
    # Use the report's lock, so CI reconciliation and worktree saves cannot race
    # a new report's collection. Release it before triage takes the issue lock.
    memory_errors: list[str] = []
    if not args.dry_run:
        with lock.exclusive(
            schedule.state_dir() / "schedule.lock",
            timeout=_env_int("OPS_REPORT_LOCK_TIMEOUT", 600),
        ):
            store = memory_pr.Store(args.repo_root.resolve())
            if store.records:
                memory_errors = store.sync(
                    collect.collect_report_history(args.repo_root.resolve()), run_dir
                )
    # Delivery and annotation must not interleave with another triage run.
    with _issue_lock():
        result = triage.triage(
            repo_root=args.repo_root.resolve(), work_dir=run_dir, dry_run=args.dry_run
        )
    if not result.errors:
        decisions = triage.collect_decisions(
            args.repo_root.resolve(),
            collect.collect_report_history(args.repo_root.resolve()),
        )
        (run_dir / "decisions.json").write_text(
            json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = {
        "pending": result.pending,
        "received": result.received,
        "unassigned": result.unassigned,
        "linked_existing": result.linked_existing,
        "errors": result.errors,
        "memory_errors": memory_errors,
    }
    (run_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if result.errors or memory_errors else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "report":
        signal.signal(signal.SIGTERM, _interrupt)
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
