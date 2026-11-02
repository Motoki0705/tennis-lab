"""Run the manual census entry point with a tiny git repo and fake agy only."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/github_actions"))
census = importlib.import_module("agy_command_census")


def test_deny_hook_never_grants_unlisted_shell_forms(tmp_path: Path) -> None:
    work, control = tmp_path / "work", tmp_path / "control"
    (work / ".agents").mkdir(parents=True)
    control.mkdir()
    policy = census.agy_api.readonly_settings()
    (control / "settings.json").write_text(json.dumps(policy))
    census.install_deny_hook(work, control)
    for command, expected in [
        ("cat a", "allow"),
        ("python3 -c pass", "deny"),
        ("cat a | head -n 1", "deny"),
    ]:
        result = subprocess.run(
            [
                sys.executable,
                str(work / ".agents/agy_census_deny.py"),
            ],
            input=json.dumps(
                {"stepIdx": 2, "toolCall": {"args": {"CommandLine": command}}}
            ),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        assert json.loads(result.stdout)["decision"] == expected


@pytest.mark.parametrize(
    "settings",
    [
        {"permissions": {"allow": ["command(*)"]}},
        {"toolPermission": "always-proceed"},
        {"trustedWorkspaces": ["~"]},
    ],
)
def test_preflight_rejects_permission_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: dict[str, object],
) -> None:
    profile = tmp_path / "profile"
    config = profile / ".gemini/antigravity-cli/settings.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps(settings))
    monkeypatch.setenv("HOME", str(profile))
    with pytest.raises(ValueError):
        census.preflight(profile / "work")


@pytest.mark.parametrize(
    "phase,timeout,settings_change",
    [
        ("pilot", False, False),
        ("pilot", True, False),
        ("measurement", False, False),
        ("pilot", False, True),
    ],
)
def test_fake_run_is_private_pinned_and_retains_partial_stream(
    tmp_path: Path,
    phase: str,
    timeout: bool,
    settings_change: bool,
) -> None:
    source = tmp_path / "source"
    assets = source / ".agents/ops/issue_rewrite"
    (assets / "prompts").mkdir(parents=True)
    (assets / "schemas").mkdir()
    template = "Production template fixture\n" + (
        "長い入力\n" * 12000 if phase == "measurement" else ""
    )
    (assets / "prompts/writer.md").write_text(template)
    (assets / "schemas/writer.schema.json").write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["body"],
                "properties": {"body": {"type": "string"}},
            }
        )
    )
    git_env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    for args in [
        ["init", "--quiet", "--template="],
        ["add", "."],
        [
            "-c",
            "user.name=Census",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
    ]:
        subprocess.run(["git", *args], cwd=source, env=git_env, check=True)
    profile = tmp_path / "profile"
    settings = profile / ".gemini/antigravity-cli/settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text('{"agentMode":"auto"}')
    private = profile / ".local/state/tennis-lab-agents/manual-runs/1566/agy-census"
    private.mkdir(parents=True)
    old_ledger = private / "invocations.jsonl"
    if phase == "measurement":
        old_ledger.write_text('{"previous":"pilot"}\n' * 25)
    phase_root = private if phase == "pilot" else private / phase
    inputs = private / "input.json"
    inputs.write_text('{"issue_number":12}')
    fake = tmp_path / "fake-agy"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    gh_calls = tmp_path / "gh-calls.jsonl"
    fake_gh.write_text(
        f"#!{sys.executable}\nimport sys,json\n"
        f"with open({str(gh_calls)!r},'a') as handle: handle.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "sys.exit(86)\n"
    )
    fake_gh.chmod(0o700)
    fake.write_text(f"""#!{sys.executable}
import json,sys,signal
from pathlib import Path
if '--version' in sys.argv:
    print('1.4.11');sys.exit(0)
args=sys.argv[1:]
Path({str(private / "argv.json")!r}).write_text(json.dumps(args))
if '--input-format' in args:
    payloads=[json.loads(line) for line in sys.stdin.read().splitlines()]
    assert len(payloads)==1 and payloads[0]['event']=='user'
    Path({str(private / "received-prompt.txt")!r}).write_text(payloads[0]['message']['content'])
Path(args[args.index('--log-file')+1]).write_text('applying agent mode plan\\nPropagating selected model override to backend: label="Gemini 3.8 Flash (High)"\\n')
print(json.dumps({{"event":"init","conversation_id":"test","init":{{"permission_mode":"request-review"}}}}),flush=True)
print(json.dumps({{"event":"step_update","step_update":{{"step_index":2,"state":"ACTIVE","tool_info":{{"name":"run_command","parameters":{{"CommandLine":"cat a"}}}}}}}}),flush=True)
if {timeout!r}: signal.pause()
if {settings_change!r}: Path({str(settings)!r}).write_text('{{"agentMode":"auto","model":"new preference"}}')
print(json.dumps({{"event":"result","result":{{"status":"SUCCESS","structured_output":{{"body":"ok"}}}}}}),flush=True)
""")
    fake.chmod(0o700)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/github_actions/agy_command_census.py"),
            "run",
            "--id",
            "fake",
            "--source",
            str(source),
            "--agy",
            str(fake),
            "--role",
            "writer",
            "--input-json",
            str(inputs),
            "--timeout",
            "600" if phase == "measurement" else "0.5" if timeout else "10",
            "--phase",
            phase,
            *(["--deny-unlisted", "--prompt-stdin"] if phase == "measurement" else []),
        ],
        env={
            **os.environ,
            "HOME": str(profile),
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == (1 if settings_change else 0), result.stderr
    capture = json.loads((phase_root / "runs/fake/control/capture.json").read_text())
    assert capture["timed_out"] == timeout
    assert capture["structured_valid"] != timeout
    assert capture["commands"][0]["command"] == "cat a"
    assert capture["commands"][0]["outcome"] == "unknown"
    assert settings.read_text() == (
        '{"agentMode":"auto","model":"new preference"}'
        if settings_change
        else '{"agentMode":"auto"}'
    )
    assert capture["persistent_settings_unchanged"] != settings_change
    assert (
        phase_root / "runs/fake/control/saved-settings-before.json"
    ).read_text() == '{"agentMode":"auto"}'
    if settings_change:
        assert "persistent agy settings changed" in result.stderr
    args = json.loads((private / "argv.json").read_text())
    policy = json.loads(args[args.index("--settings") + 1])
    assert policy["agentMode"] == args[args.index("--mode") + 1] == "plan"
    assert policy["toolPermission"] == "request-review"
    assert policy["enableTerminalSandbox"] is False
    assert all(
        rule.startswith("command(regex:^") for rule in policy["permissions"]["allow"]
    )
    assert len((phase_root / "invocations.jsonl").read_text().splitlines()) == 1
    if phase == "measurement":
        assert old_ledger.read_text() == '{"previous":"pilot"}\n' * 25
        assert "-p" not in args
        assert args[args.index("--input-format") + 1] == "stream-json"
        assert (
            private / "received-prompt.txt"
        ).read_text() == template + '\n\n入力（JSON）:\n{"issue_number": 12}'
        assert (private / "received-prompt.txt").stat().st_size > 128 * 1024
    assert capture["within_300_seconds"] is True
    assert capture["phase"] == phase
    assert not gh_calls.exists(), (
        "the harness must not invoke gh, including write operations"
    )
    assert (
        phase_root / "runs/fake/control/stdout.jsonl"
    ).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("phase,limit", [("pilot", 25), ("measurement", 60)])
def test_phase_cap_stops_before_invoking_any_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, limit: int
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = census.private_root()
    phase_root = root if phase == "pilot" else root / phase
    phase_root.mkdir(parents=True)
    ledger = phase_root / "invocations.jsonl"
    previous = '{"run":"already-recorded"}\n' * limit
    ledger.write_text(previous)
    with pytest.raises(ValueError, match=f"{limit} invocation cap"):
        census.run_turn(
            run_id="must-not-start",
            source=tmp_path,
            ref="HEAD",
            agy=tmp_path / "nonexistent-agy",
            inputs=tmp_path / "missing-input",
            role="writer",
            timeout=600,
            deny_unlisted=True,
            phase=phase,
        )
    assert ledger.read_text() == previous
    assert not (phase_root / "runs").exists()
