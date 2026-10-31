"""Evaluate emitted permission regexes as full matches and partial searches."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops/lib"))
agy: Any = importlib.import_module("agy_headless")


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
