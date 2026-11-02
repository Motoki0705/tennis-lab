"""Evaluate emitted permission regexes as full matches and partial searches."""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops/lib"))
agy: Any = importlib.import_module("agy_headless")
hook: Any = importlib.import_module("agy_hook")
policy: Any = importlib.import_module("agy_permissions")


@pytest.fixture(scope="module")
def rules() -> dict[str, list[re.Pattern[str]]]:
    permissions = agy.readonly_settings()["permissions"]
    return {
        key: [
            re.compile(rule[len("command(regex:") : -1])
            for rule in permissions[key]
            if rule.startswith("command(regex:")
        ]
        for key in ("allow", "deny")
    }


def accepted(
    rules: dict[str, list[re.Pattern[str]]], command: str, *, search: bool
) -> bool:
    def matches(pattern: re.Pattern[str]) -> bool:
        return (
            pattern.search(command) if search else pattern.fullmatch(command)
        ) is not None

    return not any(matches(rule) for rule in rules["deny"]) and any(
        matches(rule) for rule in rules["allow"]
    )


@pytest.mark.parametrize(
    "command",
    [
        "cat a",
        "cat -n README.md",
        "cat 'two words.md'",
        'cat "日本語.md"',
        "ls",
        "ls -l -a src",
        "head -n 10 a",
        "head -n10 a",
        "tail -n 20 a",
        "wc -l a",
        "grep -n -F key README.md",
        "rg -n -i key src",
        "rg -l -F key src",
        "rg -w key src",
        "rg -C 2 key src",
        "rg -A2 key src",
        "rg -B 3 key src",
        "rg -g '*.py' --hidden key src",
        'rg --glob="*.py" key src',
        "rg --files",
        "rg --files -g '*.md' .agents",
        "rg '日本語 の語' src",
        "git log --oneline -n 10",
        "git log --follow -- README.md",
        "git log --no-ext-diff --no-textconv -p --stat -n1",
        "git show --no-ext-diff --no-textconv HEAD:README.md",
        "git diff --no-ext-diff --no-textconv HEAD~1 HEAD -- src",
        "git --no-pager diff --no-ext-diff --no-textconv --cached --stat",
        "git grep -n -F key -- src",
        "gh issue view 1566 --comments",
        "gh issue view 1566 --json title,body",
        "gh issue list --limit 5 --state open",
        "gh issue list --repo Motoki0705/tennis-lab --json number,title",
        "gh pr view 1601 --comments",
        'gh pr view 1601 --json="title,body"',
    ],
)
def test_supported_read_forms(
    rules: dict[str, list[re.Pattern[str]]], command: str
) -> None:
    assert accepted(rules, command, search=False)
    assert accepted(rules, command, search=True)
    assert hook.evaluate(json_input(command))["decision"] == "allow"


@pytest.mark.parametrize(
    "command",
    [
        "cat a > b",
        "cat a >> b",
        "ls; rm -rf x",
        "ls && touch x",
        "ls || touch x",
        "cat a | tee b",
        "cat $(touch x)",
        "cat `touch x`",
        "cat <(touch x)",
        "cat a\nls",
        "cat a\n",
        "cat a\r\n",
        "cat\ta",
        "cat a\x00",
        "cat a\\\nls",
        "cat a\\ b",
        "cat {a,b}",
        "cat $FILE",
        "cat *",
        "cat ~/a",
        "cat '[x];touch x'",
        'rg "a|b" src',
        "rg '$(touch x)' src",
        "rg 'a\\b' src",
        "touch x; cat a",
        "echo cat a",
        "env cat a",
        "nice cat a",
        "cat a 2> b",
    ],
)
def test_shell_syntax_is_denied(
    rules: dict[str, list[re.Pattern[str]]], command: str
) -> None:
    assert not accepted(rules, command, search=False)
    assert not accepted(rules, command, search=True)
    assert hook.evaluate(json_input(command))["decision"] == "deny"


@pytest.mark.parametrize(
    "command",
    [
        "git diff --output=x",
        "git log --output x",
        "git -c core.pager=sh log",
        "rg --pre=sh x",
        "rg --pre sh x",
        "rg --pre-glob '*.py' x",
        "rg --pre-glob=py x",
        "git grep -Osh x",
        "git grep -O sh x",
        "git grep --open-files-in-pager=sh x",
        "git grep --open-files-in-pager sh x",
        "git diff --ext-diff",
        "git show --textconv",
        "gh issue view 1 --web",
        "gh issue list --jq .",
        "gh issue list --template x",
        "git --exec-path=x log",
        "git --git-dir=x log",
        "git --work-tree=x log",
        "git diff --no-ext-diff --no-textconv --output=x",
        "git diff --no-ext-diff --no-textconv --output x",
        "git diff --no-ext-diff --no-textconv -o x",
        "git log --no-ext-diff --no-textconv --output=x",
        "git show --no-ext-diff --no-textconv --textconv",
        "git diff --no-ext-diff --no-textconv --ext-diff",
        "git log -p",
        "git show HEAD",
        "git diff HEAD",
        "tail -f a",
        "tail --follow a",
        "cat --output=x a",
        "ls -o a",
        "rg '--pre=sh' x",
        "rg -g --pre=sh x",
        "head -n --output=x a",
        "gh issue view --json --web 1",
        "touch x",
        "rm x",
        "mkdir x",
        "cp a x",
        "mv a x",
        "python3 -c pass",
        "sh -c true",
    ],
)
def test_unlisted_flags_are_denied(
    rules: dict[str, list[re.Pattern[str]]], command: str
) -> None:
    assert not accepted(rules, command, search=False)
    assert not accepted(rules, command, search=True)
    assert hook.evaluate(json_input(command))["decision"] == "deny"


def json_input(command: str) -> str:
    return json.dumps(
        {
            "stepIdx": 7,
            "toolCall": {"name": "run_command", "args": {"CommandLine": command}},
        }
    )


@pytest.mark.parametrize(
    "command",
    [
        "git grep -nwi --column --cached --max-depth=10 -A2 -e key -- src",
        "git grep --files-without-match --untracked --no-color -e key .agents",
        "git status -sbz --porcelain=v2 --untracked-files=all --ignored=matching -- src",
        "git status --branch --short --ignore-submodules=all",
        "git log --all --full-history --grep=key -n20 --oneline -- src",
        "git log -1 --format='%h %s' --date=iso-strict --reverse HEAD~1..HEAD",
        "git log --no-ext-diff --no-textconv --name-status --stat --max-count=2",
        "ls -lahR --sort=extension --time-style=long-iso --color=never .agents src",
        "ls -AdF --group-directories-first --reverse .",
        "rg -nwi --column --max-count=100 --max-depth 5 --iglob '*.md' key .agents",
        "rg --files --hidden --no-config --type py --max-filesize=10000000 src",
        "rg --count-matches --stats --no-ignore-vcs -T md key src",
        "grep -rni --binary-files=without-match --include='*.py' -m5 key src",
        "grep --files-without-match -IC2 -e key -- src/a.py src/b.py",
        "gh issue view 1644 -c -R Motoki0705/tennis-lab --json number,title,body",
        "find -P . src -maxdepth 10 -mindepth 1 -type f -iname '*.md' -print0",
        "find . -name '*.py' -o -path './tests/*' -prune",
        "find src -size +1024c -mmin -60 -not -empty -ls",
        "git ls-files -cmz --deduplicate --exclude-standard -- src .agents",
        "git ls-files --others --ignored --exclude-standard --full-name 'src/*.py'",
        "git log --max-count=10000 --skip=0 --since=2026-01-01 --until '2 weeks ago'",
        "git grep --max-depth 100 --max-count 0 -- key src",
        "grep -A10000 -B0 -- key src/a.py",
        "find . -maxdepth 100 -size 10000000c -mtime +10000 -print",
        "gh issue list --limit=1000 --state=closed --json=number,title",
        "gh issue view https://github.com/Motoki0705/tennis-lab/issues/1644 --comments",
        "gh pr view https://github.com/Motoki0705/tennis-lab/pull/1652 --json reviews,body",
        "gh pr view codex/topic --json headRefName",
    ],
)
def test_unobserved_readonly_combinations(
    rules: dict[str, list[re.Pattern[str]]], command: str
) -> None:
    assert accepted(rules, command, search=False)
    assert accepted(rules, command, search=True)
    assert hook.evaluate(json_input(command))["decision"] == "allow"


@pytest.mark.parametrize(
    "command",
    [
        "git log -n10001",
        "git log -10001",
        "git log --max-count=10001",
        "git log --skip=-1",
        "git log --date=format:code",
        "git log --pretty=custom",
        "git log --decorate=unknown",
        "git log --format=%G?",
        "git log --format='%h %GS'",
        "git log --format='%+G?'",
        "git grep --max-depth=101 key",
        "git grep -m10001 key",
        "rg -A10001 key",
        "rg --max-depth 101 key",
        "rg --max-filesize=10000001 key",
        "rg --max-filesize=1G key",
        "rg --max-count 1.5 key",
        "rg --color=always key",
        "grep -m-1 key a",
        "grep --binary-files=text key a",
        "find . -maxdepth 101",
        "find . -size +10000001c",
        "find . -mtime +10001",
        "find . -type x",
        "ls --sort=code",
        "ls --time-style=+code",
        "gh issue list --limit 1001",
        "gh issue view ../1",
        "gh issue view 0",
        "gh issue view 1 --json unknown_field",
        "gh issue view 1 --json reviews",
        "gh issue view https://example.invalid/owner/repo/issues/1",
    ],
)
def test_values_are_typed_and_bounded(
    rules: dict[str, list[re.Pattern[str]]], command: str
) -> None:
    assert not accepted(rules, command, search=True)
    assert hook.evaluate(json_input(command))["decision"] == "deny"


@pytest.mark.parametrize(
    "command",
    [
        "cat /etc/passwd",
        "cat '../secret'",
        'cat "src/../../secret"',
        "head -n10 /etc/passwd",
        "tail ../secret",
        "wc src/../secret",
        "ls -la /tmp",
        "ls src/..",
        "ls ./../",
        "ls '..'",
        "rg key ../secret",
        "rg --files /etc",
        "rg key 'src/../secret'",
        "grep key /etc/passwd",
        "grep -rn key ../",
        "git status -- /tmp",
        "git ls-files ../",
        "git ls-files ':(top)../secret'",
        "git grep key -- /etc",
        "git grep key '../secret'",
        "git log -- ../",
        "git log /etc",
        "git log --since ../secret",
        "git show --no-ext-diff --no-textconv HEAD:../secret",
        "git diff --no-ext-diff --no-textconv HEAD -- /etc",
        "find /tmp -type f",
        "find . ../ -print",
        "find . -path ../secret",
        "rg -- --glob /etc",
        "git grep -- -e /etc/passwd",
        "grep -e key -- /etc/passwd",
        "git log -- HEAD:../secret",
        "find . -name *.py",
    ],
)
def test_path_operands_cannot_spell_an_escape(
    rules: dict[str, list[re.Pattern[str]]], command: str
) -> None:
    assert not accepted(rules, command, search=True)
    assert hook.evaluate(json_input(command))["decision"] == "deny"


@pytest.mark.parametrize(
    "command",
    [
        "git status --config-env=core.fsmonitor=EVIL",
        "git --paginate log",
        "git log --show-signature",
        "git grep --textconv key",
        "git grep --recurse-submodules key",
        "git ls-files --exclude-from=/etc/passwd",
        "git ls-files --recurse-submodules",
        "git ls-files --format=%f",
        "find . -exec touch x",
        "find . -execdir touch x",
        "find . -ok touch x",
        "find . -okdir touch x",
        "find . -delete",
        "find . -fprint out",
        "find . -fprint0 out",
        "find . -fprintf out x",
        "find . -fls out",
        "find . -files0-from a",
        "find -L .",
        "find -H .",
        "find . -follow",
        "rg -z key",
        "rg --search-zip key",
        "rg --hostname-bin sh key",
        "rg -L key",
        "rg --follow key",
        "rg --type-add py:script key",
        "rg --type-clear py key",
        "grep -Rn key .",
        "grep -f a .",
        "grep --file=a .",
        "grep --devices=read key a",
        "ls -laL .",
        "ls --dereference .",
        "ls -H .",
        "ls --dereference-command-line .",
        "ls --dereference-command-line-symlink-to-dir .",
        "gh pr view 1 -w",
        "gh issue view 1 -q .",
        "gh issue view 1 -t x",
        "pytest",
        "git status --short; touch x",
        "ls -la > out",
        "find . -print | sh",
    ],
)
def test_dangerous_tool_flags_remain_denied(
    rules: dict[str, list[re.Pattern[str]]], command: str
) -> None:
    assert not accepted(rules, command, search=True)
    assert hook.evaluate(json_input(command))["decision"] == "deny"


def test_smoke_negative_corpus_is_still_denied(tmp_path: Path) -> None:
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / "scripts/github_actions")
    )
    probe = importlib.import_module("agy_permission_probe")
    commands = probe._denial_commands(tmp_path / "outside")
    assert len(commands) == 27
    for name, command in commands:
        assert not policy.command_allowed(command), name
        assert hook.evaluate(json_input(command))["decision"] == "deny", name


def test_generated_permission_table_and_regex_contract() -> None:
    readme = Path(policy.__file__).with_name("README.md").read_text()
    table = readme.split("<!-- agy-permissions:start -->\n", 1)[1].split(
        "\n<!-- agy-permissions:end -->", 1
    )[0]
    assert table == policy.permission_table()
    assert all(
        "(?=" not in rule and "(?!" not in rule and "(?<" not in rule
        for rules in policy.command_permissions().values()
        for rule in rules
    )
    # Linux limits a single execve argument to 32 pages, independently of ARG_MAX.
    assert len(json.dumps(agy.readonly_settings()).encode()) < 128 * 1024


def test_every_allow_is_anchored_and_non_command_actions_are_denied() -> None:
    settings = agy.readonly_settings()
    assert settings["toolPermission"] == "request-review"
    assert settings["agentMode"] == "plan"
    assert settings["allowNonWorkspaceAccess"] is False
    assert settings["enableTerminalSandbox"] is False
    permissions = settings["permissions"]
    assert all(
        rule.startswith("command(regex:^") and rule.endswith("$)")
        for rule in permissions["allow"]
    )
    assert {
        "write_file(*)",
        "read_url(*)",
        "execute_url(*)",
        "unsandboxed(*)",
        "mcp(*)",
    } <= set(permissions["deny"])
    assert not any(rule.startswith("edit_file(") for rule in permissions["deny"])
