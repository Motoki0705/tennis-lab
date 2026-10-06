"""Exercise ARIS helpers with disposable repositories and CPU-only queue jobs."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = tomllib.loads((ROOT / ".agents/aris/upstream.toml").read_text())


def run(
    cwd: Path,
    *args: str,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=cwd, env=env, capture_output=True, text=True, check=check, timeout=30
    )


@pytest.fixture
def repositories(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    repository = tmp_path / "repository with spaces"
    repository.mkdir()
    shutil.copytree(ROOT / ".agents", repository / ".agents")
    run(repository, "git", "init", "-q", "-b", "main")
    run(repository, "git", "add", ".agents")
    run(
        repository,
        "git",
        "-c",
        "user.name=ARIS fixture",
        "-c",
        "user.email=aris-fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "Install fixture skills",
    )
    worktree = repository / ".claude/worktrees/campaign"
    run(repository, "git", "worktree", "add", "-q", "-b", "campaign", str(worktree))
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("TRAINING_QUEUE_") and key != "KNOWLEDGE_DIR"
    }
    environment.update(
        {
            "TRAINING_QUEUE_LOCK_FILE": str(tmp_path / "isolated-gpu.lock"),
            "TRAINING_QUEUE_SYSTEM_LOCK_FILE": str(tmp_path / "no-production-lock"),
            "TRAINING_QUEUE_PYTHON": sys.executable,
            "KNOWLEDGE_DIR": str(worktree / "knowledge"),
        }
    )
    return repository, worktree, environment


def aris(
    checkout: Path,
    environment: dict[str, str],
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return run(
        checkout,
        sys.executable,
        str(checkout / ".agents/aris/aris.py"),
        *args,
        env=environment,
        check=check,
    )


def test_installed_skills_have_metadata_dependencies_and_resolvable_links() -> None:
    installed = set(INVENTORY["skills"])
    catalog = ROOT / ".agents/aris/tools/skill-groups.tsv"
    for line in catalog.read_text().splitlines():
        fields = line.split("\t")
        if fields[0] == "skill" and fields[1] in installed and fields[3] != "-":
            assert set(fields[3].split(",")) <= installed
    for name in installed:
        path = ROOT / ".agents/skills" / name / "SKILL.md"
        metadata = yaml.safe_load(path.read_text().split("---", 2)[1])
        assert metadata["name"] == name
        assert (
            isinstance(metadata["description"], str) and metadata["description"].strip()
        )
        # Exclude Markdown syntax demonstrated inside code fences.
        prose = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
        prose = re.sub(r"`[^`]*`", "", prose)
        for target in re.findall(r"\]\(([^)]+)\)", prose):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            assert (path.parent / target.split("#", 1)[0]).exists(), (name, target)


def test_setup_resolves_shared_queue_and_rebuilds_stale_checkout_pointer(
    repositories: tuple[Path, Path, dict[str, str]],
) -> None:
    repository, worktree, environment = repositories
    before = aris(worktree, environment, "doctor", check=False)
    assert before.returncode == 1
    report = json.loads(aris(worktree, environment, "setup").stdout)
    assert report["ok"]
    assert report["paths"]["queue"] == str(repository / ".training_queue")
    assert report["paths"]["checkout"] == str(worktree)
    manifest = worktree / ".aris/installed-skills-codex.txt"
    manifest.write_text(manifest.read_text().replace(str(worktree), "/old/checkout"))
    assert aris(worktree, environment, "doctor", check=False).returncode == 1
    assert json.loads(aris(worktree, environment, "setup").stdout)["ok"]
    assert not (repository / ".training_queue").exists()


def test_queue_rejects_worktree_local_state_and_preserves_cli_failures(
    repositories: tuple[Path, Path, dict[str, str]],
) -> None:
    repository, worktree, environment = repositories
    wrong = {**environment, "TRAINING_QUEUE_DIR": str(worktree / ".training_queue")}
    result = aris(worktree, wrong, "queue", "add", "true", check=False)
    assert result.returncode == 2
    assert "shared repository queue" in result.stderr
    assert not (worktree / ".training_queue").exists()
    assert not (repository / ".training_queue").exists()
    result = aris(
        worktree,
        environment,
        "queue",
        "add",
        "true",
        "--resource",
        "invalid",
        check=False,
    )
    assert result.returncode == 2
    assert "--resource must be half or all" in result.stdout + result.stderr


def test_queue_to_knowledge_and_html_report_uses_recorded_cpu_fixture_metrics(
    repositories: tuple[Path, Path, dict[str, str]],
) -> None:
    repository, worktree, environment = repositories
    aris(worktree, environment, "setup")
    probe = worktree / "probe.py"
    probe.write_text(
        "import json, os\nfrom pathlib import Path\n"
        "root = Path(os.environ['TENNIS_REPRO_DIR']) / 'predictions'\n"
        "root.mkdir(parents=True)\n"
        "(root / 'metrics.json').write_text(json.dumps({'fixture_score': 0.75}))\n"
        "Path('executed-cwd.txt').write_text(str(Path.cwd()))\n"
    )
    command = shlex.join([sys.executable, str(probe)])
    added = aris(
        worktree,
        environment,
        "queue",
        "add",
        command,
        "--name",
        "aris-cpu-fixture",
        "--resource",
        "all",
        "--provider",
        "codex",
        "--session",
        "fixture-session",
    )
    job_file = added.stdout.strip().removeprefix("queued: ")
    queue = repository / ".training_queue"
    assert (queue / "jobs" / job_file).is_file()
    aris(worktree, environment, "queue", "serve", "--idle-timeout", "0")
    assert (queue / "done" / job_file).is_file()
    assert (worktree / "executed-cwd.txt").read_text() == str(worktree)
    assert not (worktree / ".training_queue").exists()
    repro = queue / "repro" / Path(job_file).stem
    captured = json.loads((repro / "run.json").read_text())
    assert captured["cwd"] == str(worktree)
    assert captured["command"] == command
    assert captured["session"] == "fixture-session"

    # Promote the exact queued bundle, rather than using a guessed/same-name run.
    run(
        worktree,
        sys.executable,
        str(worktree / ".agents/skills/knowledge-control/scripts/kg_register.py"),
        "--repro-dir",
        str(repro),
        "--task",
        "synthetic_data_generation",
        "--id",
        "run-aris-cpu-fixture",
        env={**environment, "TRAINING_QUEUE_DIR": str(queue)},
    )
    node = next((worktree / "knowledge/nodes").rglob("*-run-aris-cpu-fixture.md"))
    metadata = yaml.safe_load(node.read_text().split("---", 2)[1])
    assert metadata["metrics"] == {"fixture_score": 0.75}
    assert metadata["status"] == "done"
    assert metadata["provider"] == "codex"

    # Test the actual upstream renderer; no model calls or independent reviews.
    campaign = worktree / "outputs/aris/fixture"
    campaign.mkdir(parents=True)
    report = campaign / "NARRATIVE_REPORT.md"
    report.write_text(
        "# CPU接続検証\n\n隔離fixtureであり、SfMの精度実験ではない。\n\n"
        "| 指標 | 実測値 |\n|---|---:|\n| fixture_score | 0.75 |\n\n"
        "実験ID: run-aris-cpu-fixture。独立レビューは未実施。\n"
    )
    run(
        worktree,
        sys.executable,
        str(worktree / ".agents/skills/render-html/scripts/render_html.py"),
        str(report),
        "--offline",
        "--lang",
        "ja",
        env=environment,
    )
    html = report.with_suffix(".html").read_text()
    assert 'lang="ja"' in html
    assert "0.75" in html and "run-aris-cpu-fixture" in html


def test_failed_queue_job_is_not_reported_as_success(
    repositories: tuple[Path, Path, dict[str, str]],
) -> None:
    repository, worktree, environment = repositories
    command = shlex.join([sys.executable, "-c", "raise SystemExit(7)"])
    result = aris(
        worktree,
        environment,
        "queue",
        "add",
        command,
        "--name",
        "aris-failed-fixture",
        "--resource",
        "all",
        "--provider",
        "codex",
        "--session",
        "fixture-session",
    )
    job = result.stdout.strip().removeprefix("queued: ")
    aris(worktree, environment, "queue", "serve", "--idle-timeout", "0")
    queue = repository / ".training_queue"
    assert (queue / "failed" / job).is_file()
    assert not (queue / "done" / job).exists()
    assert not (queue / "repro" / Path(job).stem / "predictions/metrics.json").exists()
    run(
        worktree,
        sys.executable,
        str(worktree / ".agents/skills/knowledge-control/scripts/kg_register.py"),
        "--repro-dir",
        str(queue / "repro" / Path(job).stem),
        "--task",
        "synthetic_data_generation",
        "--id",
        "run-aris-failed-fixture",
        "--status",
        "failed",
        env={**environment, "TRAINING_QUEUE_DIR": str(queue)},
    )
    node = next((worktree / "knowledge/nodes").rglob("*-run-aris-failed-fixture.md"))
    metadata = yaml.safe_load(node.read_text().split("---", 2)[1])
    assert metadata["status"] == "failed"
    assert metadata["metrics"] == {}


def test_upstream_state_resumes_without_repeating_completed_phase(
    tmp_path: Path,
) -> None:
    helper = ROOT / ".agents/aris/tools/run_state.py"
    run(
        ROOT,
        sys.executable,
        str(helper),
        "start",
        str(tmp_path),
        "fixture",
        "--phases",
        "planning,experiments,review,report",
        "--executor",
        "codex",
    )
    artifact = tmp_path / "plan.md"
    artifact.write_text("Fixture plan; no scientific results.\n")
    verdict = tmp_path / "plan-check.json"
    verdict.write_text(json.dumps({"file_exists": artifact.is_file()}))
    run(
        ROOT,
        sys.executable,
        str(helper),
        "set",
        str(tmp_path),
        "fixture",
        "planning",
        "done",
        "--artifact",
        str(artifact),
    )
    run(
        ROOT,
        sys.executable,
        str(helper),
        "accept",
        str(tmp_path),
        "fixture",
        "planning",
        "--reviewer",
        "deterministic:fixture-file-check",
        "--verdict-id",
        str(verdict),
    )
    result = run(ROOT, sys.executable, str(helper), "resume", str(tmp_path), "fixture")
    assert result.stdout.strip() == "experiments"
    run(
        ROOT,
        sys.executable,
        str(helper),
        "set",
        str(tmp_path),
        "fixture",
        "review",
        "skipped",
    )
    state = json.loads((tmp_path / ".aris/runs/fixture.json").read_text())
    assert state["phases"][2]["status"] == "skipped"
    assert state["phases"][0]["reviewer"] == "deterministic:fixture-file-check"
