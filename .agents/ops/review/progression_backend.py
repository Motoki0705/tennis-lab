"""GitHub, parent registry and creating-worker adapters for automatic progression."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from issues.cli import GitHubHTTPError
from lib.d_capacity import CapacityError
from lib.ops_review.merge_gate_policy import auto_merge, python_ci_errors
from lib.ops_review.progression import RateWait
from lib.repair_turn import parse_report
from lib.repair_turn import request as repair_request
from parent.execution import authorization
from parent.github import GitHub
from parent.policy import child_stopped, closing_issues, object_value
from parent.session import ParentSession
from parent.state import Store


class ProgressionGitHub(GitHub):
    """Conditional reads at every mutable endpoint, with caller-owned deadlines."""

    def __init__(
        self,
        repo: str,
        *,
        store: Store | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(repo, store=store)
        self.targeted_history = True
        self.clock = clock

    def get(self, endpoint: str, *, missing_parent: bool = False) -> object:
        if endpoint.partition("?")[0] in {
            f"repos/{self.repo}/issues",
            f"repos/{self.repo}/pulls",
        }:
            raise ValueError(
                "automatic progression cannot enumerate repository issues or PRs"
            )
        if missing_parent:
            # The ordinary reader distinguishes the documented "No parent
            # issue found" response from an inaccessible resource's 404.
            try:
                return super().get(endpoint, missing_parent=True)
            except GitHubHTTPError as exc:
                if exc.status not in {403, 429}:
                    raise
                # Reobserve throttling through #2174 to obtain its deadline.
        response = self.get_conditional(endpoint, clock=self.clock)
        if response.rate_limit is not None:
            raise RateWait(response.rate_limit.resume_at, response.rate_limit.reason)
        return response.body


class LiveBackend:
    def __init__(
        self,
        state: dict[str, Any],
        store: Store,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.state, self.store, self.clock = state, store, clock
        self.repo = state["request"]["repo"]
        self.parent = state["request"]["parent_issue"]
        self.key = f"{self.repo.casefold()}#{self.parent}"
        self.gh = ProgressionGitHub(self.repo, clock=clock)
        self.gh.evidence_directory = store.directory / "evidence-issues"

    def local(self, state: dict[str, Any]) -> str | None:
        if state["request"].get("kind") != "post_merge_confirmation" and not auto_merge(
            os.environ.get("AGENT_AUTO_MERGE")
        ):
            return "auto_merge_disabled"
        with self.store.consistent_snapshot():
            row = self.store.parents.get(self.key, {})
            if (
                row.get("status") != "active"
                or row.get("session") != state["owner"]
                or row.get("generation") != state["generation"]
            ):
                return "owner_changed"
        return None

    def authorize(self) -> None:
        with self.store.consistent_snapshot():
            session = ParentSession(
                self.gh, self.store, self.parent, self.state["owner"]
            )
            if authorization(session) != self.state["request"]["authorization"]:
                raise ValueError("progression execution authority changed")

    def review(self, state: dict[str, Any], directory: Path) -> dict[str, Any]:
        from lib.ops_review.parent_review_impl import run_review

        self.authorize()
        return run_review(state, directory, self.gh)

    def pr(self, number: int) -> dict[str, Any]:
        result = object_value(self.gh.get(f"repos/{self.repo}/pulls/{number}"))
        if result.get("number") != number:
            raise ValueError("PR identity differs from request")
        return result

    def checks(self, head: str) -> list[dict[str, Any]]:
        rows = []
        page = 1
        while True:
            result = object_value(
                self.gh.get(
                    f"repos/{self.repo}/commits/{head}/check-runs?check_name=Python&filter=latest&per_page=100&page={page}"
                )
            )
            batch = result.get("check_runs")
            if not isinstance(batch, list):
                raise ValueError("check-runs response has no check_runs")
            rows.extend(batch)
            if len(rows) >= result["total_count"]:
                break
            if not batch:
                raise ValueError("incomplete check-runs pagination")
            page += 1
        return rows

    def ci(self, number: int, head: str, *, main: bool = False) -> dict[str, Any]:
        checks = self.checks(head)
        if main and checks:
            # A PR run at the same SHA is not main's push CI. Select suites
            # belonging to a push run on the requested base branch.
            base = self.state["targets"][str(number)]["base_ref"]
            runs = self.runs(head, event="push", branch=base)
            suites = {r["check_suite_id"] for r in runs}
            checks = [c for c in checks if c.get("check_suite", {}).get("id") in suites]
        errors = python_ci_errors(checks, head)
        pending = {"ci_python_missing", "ci_python_pending"}
        status = (
            "success"
            if not errors
            else "pending"
            if set(errors) <= pending
            else "failure"
        )
        return {
            "status": status,
            "head": head,
            "checks": checks,
            "reasons": list(errors),
        }

    def shape(self, target: dict[str, Any], pr: dict[str, Any]) -> str | None:
        if pr.get("state") != "open" or pr.get("merged"):
            return "PR is no longer open"
        if pr.get("draft"):
            return "draft"
        if pr.get("mergeable") is not True:
            return (
                "conflict" if pr.get("mergeable") is False else "mergeability_unknown"
            )
        if (
            pr["head"]["ref"] != target["branch"]
            or pr["base"]["ref"] != target["base_ref"]
        ):
            return "PR branch or base changed"
        default_branch = pr["base"]["repo"].get("default_branch")
        if not isinstance(default_branch, str) or not default_branch:
            return "repository default branch is unknown"
        if pr["base"]["ref"] != default_branch:
            return "PR base is not the repository default branch"
        if any(
            pr[side]["repo"]["full_name"].casefold() != self.repo.casefold()
            for side in ("head", "base")
        ):
            return "PR repository changed"
        if set(target["issues"]) != closing_issues(pr.get("body") or "", self.repo):
            return "PR closing targets changed"
        return None

    def human(self, state: dict[str, Any], target: dict[str, Any]) -> str | None:
        bodies = {}
        for resource, n in [
            ("issues", self.parent),
            *[("issues", n) for n in target["issues"] if n != self.parent],
            ("pulls", target["number"]),
        ]:
            row = object_value(self.gh.get(f"repos/{self.repo}/{resource}/{n}"))
            if row.get("number") != n or not isinstance(row.get("labels"), list):
                raise ValueError("human-control target or labels are unavailable")
            if "needs-human" in {label["name"] for label in row["labels"]}:
                return f"needs-human: {resource}/{n}"
            if resource == "issues":
                bodies[n] = row.get("body") or ""
        from parent.development import check_task_adoptions

        for body in bodies.values():
            check_task_adoptions(self.gh, body)
        with self.store.consistent_snapshot():
            row = self.store.parents.get(self.key, {})
            for n in target["issues"]:
                task = row.get("tasks", {}).get(str(n), {})
                if child_stopped(bodies[n], task.get("stopped", False)):
                    return f"target_stopped: #{n}"
            worker = target.get("worker")
            if worker:
                task = row.get("tasks", {}).get(str(worker["child"]), {})
                if task.get("delegation_id") != worker["delegation_id"]:
                    return "creating_worker_changed"
        self.authorize()
        return None

    def dependency(self, target: dict[str, Any]) -> bool:
        for dependency in target["dependencies"]:
            number = dependency["issue"]
            issue = self.gh.issue(number)
            with self.store.consistent_snapshot():
                task = (
                    self.store.parents.get(self.key, {})
                    .get("tasks", {})
                    .get(str(number), {})
                )
            if (
                issue["state"] != "closed"
                or issue.get("state_reason") != "completed"
                or child_stopped(issue.get("body") or "", task.get("stopped", False))
            ):
                return False
            # A later open PR must still hold its consumers. Discover only this
            # dependency's numbered timeline, and fetch referenced PRs by number.
            prs = self.gh.pulls({number})
            if any(pr.get("state") not in {"open", "closed"} for pr in prs):
                raise ValueError("dependency PR state is unavailable")
            if any(pr["state"] == "open" for pr in prs):
                return False
            if prs and not any(
                pr.get("merged") is True
                and pr.get("base", {}).get("ref")
                and pr["base"]["ref"]
                == pr["base"].get("repo", {}).get("default_branch")
                and pr["base"]["repo"].get("full_name", "").casefold()
                == self.repo.casefold()
                and number in closing_issues(pr.get("body") or "", self.repo)
                for pr in prs
            ):
                return False
        return True

    def runs(
        self, head: str, *, event: str | None = None, branch: str | None = None
    ) -> list[dict[str, Any]]:
        endpoint = f"repos/{self.repo}/actions/runs?head_sha={head}"
        if event:
            endpoint += f"&event={event}"
        if branch:
            from urllib.parse import quote

            endpoint += "&branch=" + quote(branch, safe="")
        return self.paged(endpoint, "workflow_runs")

    def paged(self, endpoint: str, key: str) -> list[dict[str, Any]]:
        rows = []
        separator = "&" if "?" in endpoint else "?"
        for page in range(1, 10001):
            data = object_value(
                self.gh.get(f"{endpoint}{separator}per_page=100&page={page}")
            )
            batch = data.get(key)
            if not isinstance(batch, list):
                raise ValueError(f"missing {key} in GitHub response")
            rows.extend(batch)
            if len(rows) >= data["total_count"]:
                return rows
            if not batch:
                raise ValueError("incomplete GitHub pagination")
        raise ValueError("GitHub pagination exceeded limit")

    def failure(self, target: dict[str, Any], ci: dict[str, Any]) -> dict[str, Any]:
        suites = {
            c["check_suite"]["id"] for c in ci["checks"] if c.get("name") == "Python"
        }
        runs = [r for r in self.runs(ci["head"]) if r.get("check_suite_id") in suites]
        jobs = []
        for run in runs:
            endpoint = f"repos/{self.repo}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs"
            for job in self.paged(endpoint, "jobs"):
                if job.get("conclusion") in {None, "success", "skipped", "neutral"}:
                    continue
                log = self.gh.job_log(job["id"], clock=self.clock)
                if log.rate_limit:
                    raise RateWait(log.rate_limit.resume_at, log.rate_limit.reason)
                jobs.append(
                    {
                        "id": job["id"],
                        "name": job["name"],
                        "run_id": run["id"],
                        "attempt": run["run_attempt"],
                        "url": job["html_url"],
                        "steps": job.get("steps", []),
                        "log": log.body,
                        "log_unavailable_reason": log.reason,
                    }
                )
        return {
            "ci": ci,
            "failed_jobs": jobs,
            "missing_details": None
            if jobs
            else "No failed jobs found for the observed Python check suites",
        }

    def dispatcher(self) -> Any:
        from dispatcher.schedule import Schedule
        from dispatcher.state import DispatchStore
        from dispatcher.supervisor import Supervisor

        schedule = Schedule(Path(self.state["options"]["dispatcher_state_dir"]))
        store = DispatchStore(self.store.directory, schedule)
        gh = ProgressionGitHub(self.repo, store=store, clock=self.clock)
        return Supervisor(gh, store, schedule)

    def repair(
        self, state: dict[str, Any], target: dict[str, Any], attempt: dict[str, Any]
    ) -> dict[str, Any]:
        worker = target.get("worker")
        if not worker:
            return {
                "outcome": "needs_parent",
                "parent_request": "PR has no registered creating Codex worker",
            }
        details = json.dumps(attempt["details"], ensure_ascii=False, indent=2)
        if worker["kind"] == "dispatcher":
            self.dispatcher().start_repair(
                self.parent, worker["progression_id"], attempt["id"], details
            )
            return {"outcome": "running"}
        from parent.manual_workers import run_worker

        prompt = (
            self.store.directory
            / "progressions"
            / state["id"]
            / ("repair-" + attempt["id"] + ".txt")
        )
        prompt.write_text(details, encoding="utf-8")
        gh = ProgressionGitHub(self.repo, store=self.store, clock=self.clock)
        code, receipt = run_worker(
            gh,
            self.store,
            self.parent,
            state["owner"],
            child=worker["child"],
            worker="primary",
            prompt=prompt,
            resume=True,
            restart_evidence=None,
            repair=repair_request(attempt["id"], target["number"]),
        )
        if code:
            if receipt.get("launch_state") == "capacity_deferred":
                raise CapacityError(receipt["reason"])
            if receipt.get("launch_state") == "rejected_before_start":
                return {
                    "outcome": "needs_parent",
                    "parent_request": receipt.get("reason")
                    or "Repair launch was rejected before starting; inspect the admission result",
                }
            raise ValueError(f"repair worker failed: {receipt}")
        report = self.recover_repair(state, target, attempt)
        if report is None:
            raise ValueError("repair launch/termination is unconfirmed")
        return report

    def recover_repair(
        self, state: dict[str, Any], target: dict[str, Any], attempt: dict[str, Any]
    ) -> dict[str, Any] | None:
        from dispatcher.runtime import Runtime

        worker = target["worker"]
        with self.store.consistent_snapshot():
            row = self.store.parents.get(self.key, {})
            task = row.get("tasks", {}).get(str(worker["child"]), {})
            if worker["kind"] == "dispatcher":
                matches = [
                    j
                    for j in row.get("dispatch", {}).get("jobs", {}).values()
                    if j.get("repair", {}).get("repair_id") == attempt["id"]
                ]
                if len(matches) != 1:
                    return None
                # The dispatcher owns launching its durable job reservation.
                job = matches[0]
                if job["phase"] in {"reserved", "starting", "running"}:
                    return {"outcome": "running"}
                if job["phase"] != "ended":
                    return None
                directory = Path(job["directory"]) / "runtime"
                runner = Path(job["runner"])
            else:
                from parent.manual_workers import RUNNER

                saved = task.get("repair_attempts", {}).get(attempt["id"])
                if not saved:
                    return None
                if saved["launch_state"] in {
                    "reserved",
                    "cancelled_unstarted",
                    "capacity_deferred",
                }:
                    return {"outcome": "reserved"}
                command = saved["command"]
                directory = Path(command[command.index("--run-dir") + 1])
                if not directory.is_absolute():
                    directory = self.store.directory / directory
                runner = RUNNER
        proof = Runtime(runner).inspect(directory)
        if not Runtime(runner).ended(proof):
            return (
                {"outcome": "running"}
                if proof["record"].get("result") == "running"
                else None
            )
        if proof["record"].get("result") != "completed":
            raise ValueError("repair runtime did not complete")
        report = parse_report(
            (directory / "final.md").read_text(encoding="utf-8"),
            repair_id=attempt["id"],
            pr=target["number"],
        )
        if (
            report["outcome"] == "repaired"
            and self.pr(target["number"])["head"]["sha"] != report["head_sha"]
        ):
            raise ValueError("repair report head differs from pushed PR head")
        return report

    def merge(self, target: dict[str, Any], head: str) -> None:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "merge",
                str(target["number"]),
                "--repo",
                self.repo,
                "--merge",
                "--match-head-commit",
                head,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            raise ValueError("gh pr merge failed: " + result.stderr.strip())

    def issues_completed(self, target: dict[str, Any]) -> bool:
        for n in target["issues"]:
            row = self.gh.issue(n)
            if row["state"] != "closed" or row.get("state_reason") != "completed":
                return False
        return True

    def change_size(
        self, state: dict[str, Any], before: str, after: str
    ) -> dict[str, Any]:
        if before == after:
            return {
                "reviewed_head": before,
                "current_head": after,
                "files": 0,
                "additions": 0,
                "deletions": 0,
            }
        from parent.snapshot import git

        root = Path(state["checkout"])
        try:
            for sha in (before, after):
                present = subprocess.run(
                    ["git", "cat-file", "-e", sha + "^{commit}"],
                    cwd=root,
                    capture_output=True,
                    check=False,
                    timeout=60,
                )
                if present.returncode:
                    git(
                        "fetch",
                        "--no-tags",
                        "--no-write-fetch-head",
                        "origin",
                        sha,
                        cwd=root,
                    )
            raw = git(
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                "--numstat",
                "-z",
                before,
                after,
                "--",
                cwd=root,
            )
            rows = [r.split(b"\t", 2) for r in raw.split(b"\0") if r]
            binary = sum(r[0] == b"-" for r in rows)
            return {
                "reviewed_head": before,
                "current_head": after,
                "files": len(rows),
                "additions": sum(int(r[0]) for r in rows if r[0] != b"-"),
                "deletions": sum(int(r[1]) for r in rows if r[1] != b"-"),
                "binary_files": binary,
            }
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return {
                "reviewed_head": before,
                "current_head": after,
                "files": None,
                "additions": None,
                "deletions": None,
                "unavailable_reason": str(exc),
            }

    def changes(self, state: dict[str, Any]) -> dict[str, Any]:
        reviewed = state["review"].get("result", {}).get("reviewed_heads", {})
        return {
            str(t["number"]): self.change_size(
                state,
                reviewed.get(str(t["number"]), t["head_sha"]),
                self.pr(t["number"])["head"]["sha"],
            )
            for t in state["targets"].values()
        }

    def notification_changes(
        self, state: dict[str, Any], pr: int | None
    ) -> dict[str, Any]:
        """Use only heads already observed by this progression; no later GETs."""
        reviewed = state["review"].get("result", {}).get("reviewed_heads", {})
        result = {}
        for t in state["targets"].values():
            n = str(t["number"])
            if pr is not None and t["number"] != pr:
                continue
            after = t.get("merge_head") or t.get("ci_head")
            if n in reviewed and after:
                result[n] = self.change_size(state, reviewed[n], after)
            elif n in state["review"].get("changes", {}):
                result[n] = state["review"]["changes"][n]
            else:
                result[n] = {
                    "reviewed_head": reviewed.get(n),
                    "current_head": after,
                    "files": None,
                    "additions": None,
                    "deletions": None,
                    "unavailable_reason": "review or current head not yet observed",
                }
        return result

    def notify(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        from decision_logs.review_capture import progression_event
        from dispatcher.supervisor import event as dispatch_event

        progression_event(self.store.directory, state, event)
        with self.store.locked():
            row = self.store.parents.get(self.key, {})
            if (
                row.get("session") != state["owner"]
                or row.get("generation") != state["generation"]
            ):
                return  # The durable private event remains visible to the new owner.
            dispatch = row.get("dispatch")
            if not dispatch:
                return
            delivered = row.setdefault("progression_notifications", {})
            token = f"{state['id']}:{event['id']}"
            if token in delivered:
                return
            delivered[token] = dict(event)
            dispatch["wake_requested"] = True
            if dispatch["phase"] == "waiting":
                dispatch.update(
                    phase="resume_pending", reason="レビュー・merge自動進行からの通知"
                )
            dispatch_event(
                dispatch,
                "merge_progression",
                progression_id=state["id"],
                notification_id=event["id"],
                kind_detail=event["kind"],
            )
            self.store.save()
