"""Public spin ops contracts using a disposable repo and fake user services."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
SUPERVISOR = "tennis-lab-dispatch-supervisor.service"
JUDGEMENT_TIMER = "tennis-lab-dispatch-judgement.timer"
JUDGEMENT = "tennis-lab-dispatch-judgement.service"
REPORT_TIMER = "tennis-lab-ops-report.timer"
REPORT = "tennis-lab-ops-report.service"
TRIAGE_TIMER = "tennis-lab-ops-triage.timer"
TRIAGE = "tennis-lab-ops-triage.service"
CLEANUP_TIMER = "tennis-lab-cleanup.timer"
AGENT_UNITS = {
    SUPERVISOR,
    JUDGEMENT_TIMER,
    JUDGEMENT,
    REPORT_TIMER,
    REPORT,
    TRIAGE_TIMER,
    TRIAGE,
}


@dataclass
class ControlEnv:
    repo: Path
    environment: dict[str, str]
    state_file: Path
    calls_file: Path

    def read(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.state_file.read_text())
        return data

    def write(self, state: dict[str, Any]) -> None:
        self.state_file.write_text(json.dumps(state))

    def calls(self) -> list[list[str]]:
        if not self.calls_file.exists():
            return []
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()]

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "spin", "ops", *args],
            cwd=self.repo,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )


@pytest.fixture
def control_factory(tmp_path: Path) -> Callable[[str], ControlEnv]:
    def make(name: str) -> ControlEnv:
        base = tmp_path / name
        repo = base / "repo"
        ops = repo / ".agents/ops"
        shutil.copytree(
            ROOT / ".agents/ops/ops_control",
            ops / "ops_control",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        (ops / "lib").mkdir()
        shutil.copy(ROOT / ".agents/ops/lib/checkout_lock.py", ops / "lib")
        shutil.copy(ROOT / ".agents/ops/install_units.sh", ops)
        for job in (
            "dispatcher",
            "ops-report",
            "cleanup",
            "board",
            "git_sync",
            "worktree_cleanup",
        ):
            shutil.copytree(ROOT / f".agents/ops/{job}/systemd", ops / job / "systemd")
        spin = repo / ".spin"
        spin.mkdir()
        shutil.copy(ROOT / ".spin/cmds.py", spin)
        (repo / "pyproject.toml").write_text(
            '[tool.spin]\npackage = "src"\n[tool.spin.commands]\n'
            'Ops = [".spin/cmds.py:ops"]\n'
        )
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        fake = Path(__file__).with_name("fake_ops_control.py").read_text()
        (ops / "dispatcher/__main__.py").write_text(fake)
        bin_dir = base / "bin"
        bin_dir.mkdir()
        for command in ("systemctl", "journalctl"):
            executable = bin_dir / command
            executable.write_text(f"#!{sys.executable}\n{fake}")
            executable.chmod(0o700)
        environment = {
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "XDG_CONFIG_HOME": str(base / "config"),
            "XDG_STATE_HOME": str(base / "state"),
            "FAKE_OPS_STATE": str(base / "fake-state.json"),
            "FAKE_OPS_CALLS": str(base / "calls.jsonl"),
        }
        control = ControlEnv(
            repo,
            environment,
            Path(environment["FAKE_OPS_STATE"]),
            Path(environment["FAKE_OPS_CALLS"]),
        )
        units = {}
        for path in ops.glob("*/systemd/*"):
            timer = path.suffix == ".timer"
            units[path.name] = {
                "FragmentPath": str(path),
                "ExecStart": next(
                    (
                        line.removeprefix("ExecStart=")
                        for line in path.read_text().splitlines()
                        if line.startswith("ExecStart=")
                    ),
                    "",
                ).replace("%h/projects/tennis-lab", str(repo)),
                "LoadState": "loaded",
                "UnitFileState": "disabled"
                if timer or path.name == SUPERVISOR
                else "static",
                "ActiveState": "inactive",
                "SubState": "dead",
                "Result": "" if timer else "success",
                "ExecMainStatus": "0",
                "NextElapseUSecRealtime": "Thu 2026-11-12 20:00:00 JST"
                if timer
                else "",
                "NextElapseUSecMonotonic": "0",
                "LastTriggerUSec": "",
            }
        control.write(
            {
                "units": units,
                "dispatcher": {
                    "schema_version": 1,
                    "enabled": False,
                    "parents": [],
                    "active_judgement": None,
                },
            }
        )
        return control

    return make


@pytest.fixture
def control(control_factory: Callable[[str], ControlEnv]) -> ControlEnv:
    return control_factory("control")


def checked(control: ControlEnv, *args: str) -> dict[str, Any]:
    result = control.run(*args)
    assert result.returncode == 0, result.stdout + result.stderr
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def test_agents_matches_individual_persistent_on_and_complete_off(
    control_factory: Callable[[str], ControlEnv],
) -> None:
    grouped, individual = control_factory("grouped"), control_factory("individual")
    checked(grouped, "on", "agents")
    for name in ("dispatcher", "report", "triage"):
        checked(individual, "on", name)
    enabled = {
        unit
        for unit, data in grouped.read()["units"].items()
        if data["UnitFileState"] == "enabled"
    }
    assert enabled == {SUPERVISOR, JUDGEMENT_TIMER, REPORT_TIMER, TRIAGE_TIMER}
    for env in (grouped, individual):
        state = env.read()
        for unit in AGENT_UNITS:
            state["units"][unit]["ActiveState"] = "active"
        env.write(state)
    checked(grouped, "off", "agents")
    for name in ("dispatcher", "report", "triage"):
        checked(individual, "off", name)
    for env in (grouped, individual):
        assert not env.read()["dispatcher"]["enabled"]
        for unit in AGENT_UNITS:
            assert env.read()["units"][unit]["ActiveState"] == "inactive"
        assert env.read()["units"][CLEANUP_TIMER]["UnitFileState"] == "disabled"

    # Compare mutations, not reads or fixture-specific absolute source paths.
    def mutations(env: ControlEnv) -> list[list[str]]:
        return [
            [Path(arg).name if arg.startswith(str(env.repo)) else arg for arg in call]
            for call in env.calls()
            if (call[0] == "dispatcher" and call[1] != "status")
            or (call[0] == "systemctl" and call[2] != "show")
        ]

    assert mutations(grouped) == mutations(individual)
    assert ["systemctl", "--user", "link", JUDGEMENT_TIMER] in mutations(grouped)
    # Re-enabling works after disable has removed and relinked the unit source.
    checked(grouped, "on", "dispatcher")


def test_status_distinguishes_enablement_activity_result_and_schedule(
    control: ControlEnv,
) -> None:
    state = control.read()
    state["units"][REPORT_TIMER].update(UnitFileState="enabled", ActiveState="active")
    state["units"][REPORT].update(Result="exit-code", ExecMainStatus="17")
    control.write(state)
    data = checked(control, "status")
    assert {row["target"] for row in data["targets"]} == {
        "dispatcher",
        "report",
        "triage",
        "cleanup",
        "git-sync",
        "worktree-cleanup",
    }
    report = next(row for row in data["targets"] if row["target"] == "report")
    units = {unit["unit"]: unit for unit in report["units"]}
    assert units[REPORT_TIMER]["unit_file_state"] == "enabled"
    assert units[REPORT_TIMER]["active_state"] == "active"
    assert units[REPORT_TIMER]["next_elapse_realtime"] == "Thu 2026-11-12 20:00:00 JST"
    assert units[REPORT]["active_state"] == "inactive"
    assert units[REPORT]["last_result"] == "exit-code"
    assert units[REPORT]["exec_main_status"] == "17"
    assert units[REPORT]["next_elapse_realtime"] is None


@pytest.mark.parametrize("mode", ["bus_unreachable", "manager_missing"])
@pytest.mark.parametrize("action", ["status", "on", "off", "install", "run"])
def test_unavailable_manager_is_an_error_before_mutations(
    control: ControlEnv,
    mode: str,
    action: str,
) -> None:
    state = control.read()
    state["manager"] = mode
    control.write(state)
    result = control.run(action, "report")
    assert result.returncode != 0
    data = json.loads(result.stdout)
    assert data["manager"] == "unavailable"
    assert "loginctl enable-linger" in data["error"]
    assert data["targets"] == []
    assert control.calls() == [
        ["systemctl", "--user", "show", "--property=Version", "--value"]
    ]
    assert not Path(control.environment["XDG_CONFIG_HOME"]).exists()


@pytest.mark.parametrize(
    "residue",
    [
        "timer-active",
        "timer-enabled",
        "service-active",
        "supervisor-active",
        "children",
        "job",
        "occupied",
        "judgement",
        "dispatcher-enabled",
        "bad-evidence",
    ],
)
def test_off_reports_residual_units_and_unconfirmed_dispatcher_children(
    control: ControlEnv,
    residue: str,
) -> None:
    state = control.read()
    for unit in AGENT_UNITS:
        state["units"][unit]["ActiveState"] = "active"
    after: dict[str, Any] = {
        "schema_version": 1,
        "enabled": False,
        "active_judgement": None,
        "parents": [],
    }
    if residue in {
        "timer-active",
        "timer-enabled",
        "service-active",
        "supervisor-active",
    }:
        unit = {
            "timer-active": JUDGEMENT_TIMER,
            "timer-enabled": JUDGEMENT_TIMER,
            "service-active": JUDGEMENT,
            "supervisor-active": SUPERVISOR,
        }[residue]
        if residue == "timer-enabled":
            state["sticky_enabled"] = [unit]
            state["units"][unit]["UnitFileState"] = "enabled"
        else:
            state["sticky_active"] = [unit]
    elif residue in {"children", "job", "occupied", "bad-evidence"}:
        parent: dict[str, Any] = {
            "parent": "o/r#1",
            "children_pending": [],
            "jobs": [],
            "occupies_slot": False,
        }
        if residue == "children":
            parent["children_pending"] = [42]
        elif residue == "job":
            parent["jobs"] = [{"id": "job-1", "phase": "running"}]
        elif residue == "occupied":
            parent["occupies_slot"] = True
        else:
            del parent["jobs"]
        after["parents"] = [parent]
    elif residue == "judgement":
        after["active_judgement"] = "judgement-1"
    else:
        after["enabled"] = True
    state["after_shutdown"] = after
    control.write(state)
    result = control.run("off", "agents")
    assert result.returncode != 0
    data = json.loads(result.stdout)
    assert data["outcome"] == "partial_failure"
    assert data["targets"][0]["errors"]
    assert [row["outcome"] for row in data["targets"][1:]] == ["ok", "ok"]
    assert control.read()["units"][REPORT]["ActiveState"] == "inactive"
    assert control.read()["units"][CLEANUP_TIMER]["UnitFileState"] == "disabled"


def test_shutdown_failure_still_disables_supervisor_and_stops_other_targets(
    control: ControlEnv,
) -> None:
    state = control.read()
    state["dispatcher_fail"] = ["shutdown"]
    control.write(state)
    result = control.run("off", "agents")
    assert result.returncode != 0
    assert "fake dispatcher failure" in result.stdout
    calls = control.calls()
    shutdown = calls.index(["dispatcher", "shutdown"])
    assert (
        calls.index(["systemctl", "--user", "disable", "--now", SUPERVISOR]) > shutdown
    )
    assert ["systemctl", "--user", "stop", REPORT] in calls


def test_dispatcher_enable_failure_does_not_start_its_units_but_continues_group(
    control: ControlEnv,
) -> None:
    state = control.read()
    state["dispatcher_fail"] = ["enable"]
    control.write(state)
    result = control.run("on", "agents")
    assert result.returncode != 0
    data = json.loads(result.stdout)
    assert data["outcome"] == "partial_failure"
    assert data["targets"][0]["errors"]
    assert [row["outcome"] for row in data["targets"][1:]] == ["ok", "ok"]
    assert control.read()["units"][SUPERVISOR]["UnitFileState"] == "disabled"
    assert control.read()["units"][REPORT_TIMER]["UnitFileState"] == "enabled"


@pytest.mark.parametrize("target", ["triage", "agents"])
def test_manager_loss_during_mutation_stays_unavailable(
    control: ControlEnv, target: str
) -> None:
    state = control.read()
    state["lose_bus_after_mutation"] = True
    control.write(state)
    result = control.run("on", target)
    assert result.returncode != 0
    data = json.loads(result.stdout)
    assert data["manager"] == data["outcome"] == "unavailable"
    assert "loginctl enable-linger" in data["error"]
    if target == "triage":
        assert control.read()["units"][TRIAGE_TIMER]["UnitFileState"] == "enabled"
    else:
        assert control.read()["units"][SUPERVISOR]["UnitFileState"] == "enabled"
        for timer in (REPORT_TIMER, TRIAGE_TIMER):
            assert [
                "systemctl",
                "--user",
                "enable",
                "--now",
                timer,
            ] not in control.calls()
            assert control.read()["units"][timer]["UnitFileState"] == "disabled"


@pytest.mark.parametrize("loss_point", ["verification", "shutdown", "source"])
@pytest.mark.parametrize("reconnect", [True, False], ids=["transient", "persistent"])
def test_off_manager_loss_keeps_reports_and_attempts_all_stops(
    control: ControlEnv, loss_point: str, reconnect: bool
) -> None:
    state = control.read()
    for unit in AGENT_UNITS:
        state["units"][unit]["ActiveState"] = "active"
    for unit in (SUPERVISOR, JUDGEMENT_TIMER, REPORT_TIMER, TRIAGE_TIMER):
        state["units"][unit]["UnitFileState"] = "enabled"
    state["dispatcher"]["enabled"] = True
    if loss_point == "verification":
        trigger = [
            "systemctl",
            "--user",
            "link",
            str(control.repo / ".agents/ops/dispatcher/systemd" / SUPERVISOR),
        ]
    elif loss_point == "shutdown":
        trigger = ["dispatcher", "shutdown"]
    else:
        trigger = ["systemctl", "--user", "stop", JUDGEMENT]
    state["lose_bus_after_call"] = trigger
    if reconnect:
        # Fail the unit operation/read and its manager probe, then reconnect.
        state["bus_failures_remaining"] = 2
    control.write(state)
    result = control.run("off", "agents")
    calls = control.calls()
    cut = calls.index(trigger)
    for timer, service in ((REPORT_TIMER, REPORT), (TRIAGE_TIMER, TRIAGE)):
        assert calls.index(["systemctl", "--user", "disable", "--now", timer]) > cut
        assert calls.index(["systemctl", "--user", "stop", service]) > cut
    assert ["systemctl", "--user", "disable", "--now", SUPERVISOR] in calls
    assert ["systemctl", "--user", "stop", JUDGEMENT] in calls
    assert result.returncode != 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["manager"] == data["outcome"] == "unavailable"
    assert "loginctl enable-linger" in data["error"]
    assert [report["target"] for report in data["targets"]] == [
        "dispatcher",
        "report",
        "triage",
    ]
    dispatcher, *others = data["targets"]
    assert dispatcher["outcome"] == "unavailable"
    assert any("loginctl enable-linger" in error for error in dispatcher["errors"])
    assert dispatcher["dispatcher"]["enabled"] is False
    if reconnect:
        assert dispatcher["units"]  # Keep the evidence obtained after reconnection.
        assert [report["outcome"] for report in others] == ["ok", "ok"]
        for timer, service in ((REPORT_TIMER, REPORT), (TRIAGE_TIMER, TRIAGE)):
            assert control.read()["units"][timer]["UnitFileState"] != "enabled"
            assert control.read()["units"][service]["ActiveState"] == "inactive"
    else:
        assert [report["outcome"] for report in others] == [
            "unavailable",
            "unavailable",
        ]
        assert all(report["errors"] for report in others)
        assert control.read()["units"][REPORT]["ActiveState"] == "active"


def test_incomplete_unit_properties_never_imply_inactive(control: ControlEnv) -> None:
    state = control.read()
    state["omit_properties"] = ["ActiveState"]
    control.write(state)
    result = control.run("status", "report")
    assert result.returncode != 0
    assert "missing systemd properties: ActiveState" in result.stdout
    assert json.loads(result.stdout)["targets"][0]["units"] == []


@pytest.mark.parametrize(
    "target", ["report", "triage", "cleanup", "git-sync", "worktree-cleanup"]
)
def test_run_starts_only_oneshot_without_enabling_timer(
    control: ControlEnv, target: str
) -> None:
    checked(control, "run", target)
    prefix = {
        "report": "tennis-lab-ops-report",
        "triage": "tennis-lab-ops-triage",
        "cleanup": "tennis-lab-cleanup",
        "git-sync": "tennis-lab-git-sync",
        "worktree-cleanup": "tennis-lab-worktree-cleanup",
    }[target]
    assert ["systemctl", "--user", "start", prefix + ".service"] in control.calls()
    assert control.read()["units"][prefix + ".timer"]["UnitFileState"] == "disabled"
    assert not any("enable" in call for call in control.calls())


@pytest.mark.parametrize(
    "args",
    [
        ("run", "dispatcher"),
        ("run", "agents"),
        ("on",),
        ("on", "unknown"),
        ("off", "agents", "--dry-run"),
    ],
)
def test_invalid_actions_have_no_side_effects(
    control: ControlEnv, args: tuple[str, ...]
) -> None:
    assert control.run(*args).returncode == 2
    assert control.calls() == []


def test_git_sync_controls_real_registered_timer_and_stops_running_service(
    control: ControlEnv,
) -> None:
    timer, service = "tennis-lab-git-sync.timer", "tennis-lab-git-sync.service"
    checked(control, "on", "git-sync")
    state = control.read()
    assert state["units"][timer]["UnitFileState"] == "enabled"
    assert state["units"][timer]["ActiveState"] == "active"
    state["units"][service]["ActiveState"] = "active"
    control.write(state)
    checked(control, "off", "git-sync")
    state = control.read()
    assert state["units"][timer]["UnitFileState"] == "linked"
    assert state["units"][service]["ActiveState"] == "inactive"
    assert all(
        state["units"][name]["UnitFileState"] != "enabled" for name in AGENT_UNITS
    )
    source = control.repo / ".agents/ops/git_sync/systemd" / timer
    assert "OnCalendar=*-*-* *:00,30:00 Asia/Tokyo" in source.read_text()
    assert "Persistent=true" in source.read_text()


def test_install_preview_and_update_never_enable_or_start(control: ControlEnv) -> None:
    preview = control.run("install", "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert control.calls() == []
    config = Path(control.environment["XDG_CONFIG_HOME"]) / "systemd/user"
    assert not config.exists()
    result = control.run("install")
    assert result.returncode == 0, result.stderr
    for source in (control.repo / ".agents/ops").glob("*/systemd/*"):
        assert (config / source.name).resolve() == source
    unit = config / REPORT_TIMER
    unit.unlink()
    unit.symlink_to(control.repo / "obsolete-source")
    assert control.run("install").returncode == 0
    assert (
        unit.resolve() == control.repo / ".agents/ops/ops-report/systemd" / REPORT_TIMER
    )
    mutations = [call for call in control.calls() if call[2] != "show"]
    assert mutations == [["systemctl", "--user", "daemon-reload"]] * 2


def test_targeted_install_expands_agents_without_maintenance(
    control: ControlEnv,
) -> None:
    result = control.run("install", "agents")
    assert result.returncode == 0, result.stderr
    config = Path(control.environment["XDG_CONFIG_HOME"]) / "systemd/user"
    assert {path.name for path in config.iterdir()} == AGENT_UNITS


@pytest.mark.parametrize(
    "target,expected",
    [
        ("agents", AGENT_UNITS),
        ("dispatcher", {SUPERVISOR, JUDGEMENT_TIMER, JUDGEMENT}),
        ("git-sync", {"tennis-lab-git-sync.timer", "tennis-lab-git-sync.service"}),
    ],
)
def test_logs_selects_exact_user_units_even_without_manager(
    control: ControlEnv,
    target: str,
    expected: set[str],
) -> None:
    state = control.read()
    state.update(manager="bus_unreachable", journal_exit=9)
    control.write(state)
    result = control.run("logs", target, "--lines", "17", "--follow")
    assert result.returncode == 9
    (call,) = control.calls()
    assert call[:5] == ["journalctl", "--user", "--lines", "17", "--follow"]
    assert set(call[6::2]) == expected


@pytest.mark.parametrize("kind", ["timer", "service"])
def test_new_target_only_needs_registry_entry_and_unit_files(
    control: ControlEnv, kind: str
) -> None:
    ops = control.repo / ".agents/ops"
    registry = ops / "ops_control/targets.py"
    with registry.open("a") as stream:
        stream.write(
            f'\nTARGETS["future"] = {kind}_target("future", "tennis-lab-future")\n'
        )
    state = control.read()
    for suffix in ("timer", "service") if kind == "timer" else ("service",):
        source = ops / f"future/systemd/tennis-lab-future.{suffix}"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("[Unit]\n")
        state["units"][source.name] = {
            **state["units"][REPORT_TIMER if suffix == "timer" else REPORT],
            "FragmentPath": str(source),
            "UnitFileState": "disabled",
        }
    control.write(state)
    assert control.run("install", "future").returncode == 0
    checked(control, "on", "future")
    assert checked(control, "status", "future")["targets"][0]["kind"] == kind
    if kind == "timer":
        checked(control, "run", "future")
    else:
        assert control.run("run", "future").returncode == 2
    assert control.run("logs", "future").returncode == 0
    checked(control, "off", "future")
    assert not any(call[0] == "dispatcher" for call in control.calls())
    config = Path(control.environment["XDG_CONFIG_HOME"]) / "systemd/user"
    for source in (ops / "future/systemd").iterdir():
        assert (config / source.name).is_symlink()
        assert (config / source.name).resolve() == source


def test_worktree_cleanup_timer_public_controls_and_tokyo_schedule(
    control: ControlEnv,
) -> None:
    target = "worktree-cleanup"
    stem = "tennis-lab-worktree-cleanup"
    assert control.run("install", target).returncode == 0
    checked(control, "on", target)
    data = checked(control, "status", target)
    assert {unit["unit"] for unit in data["targets"][0]["units"]} == {
        stem + ".timer",
        stem + ".service",
    }
    assert control.run("logs", target).returncode == 0
    checked(control, "off", target)
    assert ["systemctl", "--user", "stop", stem + ".service"] in control.calls()
    source = control.repo / ".agents/ops/worktree_cleanup/systemd"
    timer = (source / (stem + ".timer")).read_text()
    assert "OnCalendar=*-*-* 04:30:00 Asia/Tokyo" in timer
    assert "RandomizedDelaySec" not in timer
    assert "Persistent=true" in timer
    service = (source / (stem + ".service")).read_text()
    assert "-m worktree_cleanup cleanup" in service
    assert "KillMode=control-group" in service
