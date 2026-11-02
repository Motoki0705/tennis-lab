"""One read-only argv definition for agy settings, its hook and documentation.

The regex language deliberately needs neither lookaround nor shell tokenization.
The caller must use readonly_environment: Git configuration can execute helpers
without any dangerous command-line option. This is not an OS sandbox.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

_SPACE = r"\x20+"
_UNSAFE = ";&|<>$`(){}\\"


def _excluding(characters: str) -> str:
    escaped = "".join("\\" + c if c in "\\]-^" else c for c in characters)
    return r"[^\x00-\x20\x7f" + escaped + "]"


def _literal(value: str) -> str:
    return re.escape(value).replace(r"\-", "-").replace(r"\ ", r"\x20")


def _choices(*values: str) -> str:
    return "(?:" + "|".join(_literal(value) for value in values) + ")"


def _token(bare: str, quoted: str | None = None) -> str:
    quoted = bare if quoted is None else quoted
    return "(?:" + bare + "|'" + quoted + "'|\"" + quoted + '")'


def _argument(exclude: str = "") -> str:
    unsafe = _UNSAFE + exclude
    bare = _excluding(unsafe + "'\"*?[]~-") + _excluding(unsafe + "'\"*?[]") + "*"
    quoted = (
        _excluding(unsafe + "'\"-") + "(?:" + _excluding(unsafe + "'\"") + r"|\x20)*"
    )
    return _token(bare, quoted)


def _path_raw(*, quoted: bool) -> str:
    # No absolute paths, colon (Git magic/drive letters), or components starting
    # with '..' or '-'. Quoting never disables these checks. A single '.' is OK.
    excluded = _UNSAFE + "'\"/:" + ("" if quoted else "*?[]")
    first = _excluding(excluded + ".-~")
    rest = _excluding(excluded)
    if quoted:
        rest = "(?:" + rest + r"|\x20)"
    component = r"(?:\.|\.?" + first + rest + "*)"
    return component + "(?:/" + component + ")*/?"


def _path() -> str:
    return _token(_path_raw(quoted=False), _path_raw(quoted=True))


def _history() -> str:
    # Revisions/ranges cannot smuggle a path traversal into REV:PATH. After '--'
    # only _path is used, so a revision never acts as an absolute pathspec.
    ref = r"[A-Za-z0-9_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_][A-Za-z0-9_.-]*)*"
    rev = ref + r"(?:[~^][0-9]{0,4})*"
    revision = rev + r"(?:\.\.\.?" + rev + ")?"

    def raw(quoted: bool) -> str:
        path = _path_raw(quoted=quoted)
        return "(?:" + revision + "(?::" + path + ")?|" + path + ")"

    return _token(raw(False), raw(True))


# Bounds are decimal and explicit, including attached short-option values.
_COUNT = r"(?:0|[1-9][0-9]{0,3}|10000)"
_DEPTH = r"(?:0|[1-9][0-9]?|100)"
_BYTES = r"(?:0|[1-9][0-9]{0,6}|10000000)"
_FORMATS = (
    "oneline",
    "short",
    "medium",
    "full",
    "fuller",
    "reference",
    "raw",
    "%H",
    "%h",
    "%s",
    "%b",
    "%B",
    "%an",
    "%ad",
    "%aI",
    "%cI",
    "%h %s",
    "%h %ad %s",
    "%H %s",
)
_ISSUE_FIELDS = [
    "assignees",
    "author",
    "blockedBy",
    "blocking",
    "body",
    "closed",
    "closedAt",
    "closedByPullRequestsReferences",
    "comments",
    "createdAt",
    "id",
    "isPinned",
    "issueType",
    "labels",
    "milestone",
    "number",
    "parent",
    "projectCards",
    "projectItems",
    "reactionGroups",
    "state",
    "stateReason",
    "subIssues",
    "subIssuesSummary",
    "title",
    "updatedAt",
    "url",
]
_PR_FIELDS = [
    "additions",
    "assignees",
    "author",
    "autoMergeRequest",
    "baseRefName",
    "baseRefOid",
    "body",
    "changedFiles",
    "closed",
    "closedAt",
    "closingIssuesReferences",
    "comments",
    "commits",
    "createdAt",
    "deletions",
    "files",
    "fullDatabaseId",
    "headRefName",
    "headRefOid",
    "headRepository",
    "headRepositoryOwner",
    "id",
    "isCrossRepository",
    "isDraft",
    "labels",
    "latestReviews",
    "maintainerCanModify",
    "mergeCommit",
    "mergeStateStatus",
    "mergeable",
    "mergedAt",
    "mergedBy",
    "milestone",
    "number",
    "potentialMergeCommit",
    "projectCards",
    "projectItems",
    "reactionGroups",
    "reviewDecision",
    "reviewRequests",
    "reviews",
    "state",
    "statusCheckRollup",
    "title",
    "updatedAt",
    "url",
]


def _value(kind: str) -> str:
    if kind == "argument":
        return _argument()
    if kind == "path":
        return _path()
    if kind == "history":
        return _history()
    if kind == "basename":
        return _argument("/")
    if kind in ("issue_fields", "pr_fields"):
        fields = _choices(*(_ISSUE_FIELDS if kind == "issue_fields" else _PR_FIELDS))
        return _token(fields + "(?:," + fields + ")*")
    if kind in ("issue_id", "pr_id"):
        endpoint = "issues" if kind == "issue_id" else "pull"
        return _token(
            r"(?:[1-9][0-9]{0,8}|https://github\.com/[A-Za-z0-9_][A-Za-z0-9_.-]*/"
            r"[A-Za-z0-9_][A-Za-z0-9_.-]*/" + endpoint + r"/[1-9][0-9]{0,8})"
        )
    raw = {
        "count": _COUNT,
        "depth": _DEPTH,
        "bytes": _BYTES,
        "limit": r"(?:[1-9][0-9]{0,2}|1000)",
        "id": r"[1-9][0-9]{0,8}",
        "signed_count": "[+-]?" + _COUNT,
        "size": "[+-]?" + _BYTES + "c",
        "repository": r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*",
        "state": _choices("open", "closed", "all"),
        "porcelain": _choices("v1", "v2"),
        "untracked": _choices("no", "normal", "all"),
        "ignored": _choices("no", "traditional", "matching"),
        "all": "all",
        "never": "never",
        "without_match": "without-match",
        "skip": "skip",
        "sort": _choices("none", "size", "time", "version", "extension", "width"),
        "time_style": _choices("full-iso", "long-iso", "iso", "locale"),
        "date": _choices(
            "default",
            "relative",
            "local",
            "iso",
            "iso-strict",
            "rfc",
            "short",
            "raw",
            "human",
            "unix",
        ),
        "date_filter": r"(?:[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])|[1-9][0-9]{0,3}\x20(?:day|week|month|year)s?\x20ago)",
        "format": "(?:"
        + _choices(*_FORMATS)
        + "|(?:format|tformat):"
        + _choices(*_FORMATS)
        + ")",
        "decorate": _choices("short", "full", "no", "auto"),
        "file_type": "[bcdflps]",
        "search_type": r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",
    }[kind]
    return _token(raw)


def _switches(names: str) -> dict[str, str | None]:
    return dict.fromkeys(names.split())


@dataclass(frozen=True)
class Tool:
    name: str
    shape: str
    flags: dict[str, str | None]
    excluded: str
    patch: dict[str, str | None] = field(default_factory=dict)


_SEARCH = _switches(
    "-n --line-number -i --ignore-case -F --fixed-strings -w --word-regexp "
    "-v --invert-match -l --files-with-matches -c --count -q --quiet"
)
_CONTEXT: dict[str, str | None] = {
    "-A": "count",
    "-B": "count",
    "-C": "count",
    "-m": "count",
    "--max-count": "count",
}
_GREP = {
    **_SEARCH,
    **_CONTEXT,
    **_switches(
        "-E --extended-regexp -G --basic-regexp -L --files-without-match -h -H -I"
    ),
    "-e": "argument",
}
_GH: dict[str, str | None] = {
    "--json": "issue_fields",
    "--repo": "repository",
    "-R": "repository",
}
_GH_EXCLUDED = "--web/-w, --jq/-q, --template/-t"
_PATCH = _switches("-p --stat --name-only --name-status")
_GIT_EXCLUDED = "-c/--config-env, --output/-o, pager, --ext-diff, --textconv, --show-signature, --format=%G*"

# This table is the source of truth, including aliases. '--' is a delimiter,
# never a repeatable switch: option-looking tokens after it remain operands.
TOOLS = (
    Tool(
        "git grep",
        "search",
        {
            **_GREP,
            **_switches(
                "--column --full-name --cached --untracked --no-color --no-textconv -W --function-context"
            ),
            "--max-depth": "depth",
        },
        "-O/--open-files-in-pager, --textconv, --recurse-submodules",
    ),
    Tool(
        "git status",
        "paths",
        {
            **_switches("--short -s --branch -b -z"),
            "--porcelain": "porcelain?",
            "--untracked-files": "untracked",
            "--ignored": "ignored",
            "--ignore-submodules": "all",
        },
        "-c/--config-env, pager, absolute/parent path",
    ),
    Tool(
        "git log",
        "history",
        {
            **_switches(
                "--oneline --all-match --fixed-strings --regexp-ignore-case --reverse --no-merges --merges --first-parent --follow --no-ext-diff --no-textconv --all --full-history"
            ),
            "-n": "count",
            "--max-count": "count",
            "--skip": "count",
            "--since": "date_filter",
            "--until": "date_filter",
            "--author": "argument",
            "--committer": "argument",
            "--grep": "argument",
            "--date": "date",
            "--format": "format",
            "--pretty": "format",
            "--decorate": "decorate?",
        },
        _GIT_EXCLUDED,
        _PATCH,
    ),
    Tool(
        "ls",
        "paths",
        {
            **_switches(
                "-a --all -A --almost-all -l -h --human-readable -1 -d --directory -F --classify -p -R --recursive -r --reverse -t -S -U --group-directories-first"
            ),
            "--sort": "sort",
            "--time-style": "time_style",
            "--color": "never",
        },
        "-L/--dereference, -H/--dereference-command-line, --dereference-command-line-symlink-to-dir",
    ),
    Tool(
        "rg",
        "search",
        {
            **_SEARCH,
            **_CONTEXT,
            **_switches(
                "--column -s --case-sensitive -S --smart-case --files-without-match --count-matches -o --only-matching --hidden --files --no-heading --no-config --stats --no-ignore --no-ignore-vcs"
            ),
            "--context": "count",
            "--max-depth": "depth",
            "-g": "argument",
            "--glob": "argument",
            "--iglob": "argument",
            "--type": "search_type",
            "--type-not": "search_type",
            "-t": "search_type",
            "-T": "search_type",
            "--color": "never",
            "--max-filesize": "bytes",
        },
        "--pre, --pre-glob, --hostname-bin, -z/--search-zip, -L/--follow, --type-add, --type-clear",
    ),
    Tool(
        "grep",
        "search_file",
        {
            **_GREP,
            **_switches("-o --only-matching -r --recursive"),
            "--include": "argument",
            "--exclude": "argument",
            "--exclude-dir": "argument",
            "--color": "never",
            "--binary-files": "without_match",
            "--devices": "skip",
        },
        "-R/--dereference-recursive, -f/--file, --devices=read",
    ),
    Tool("gh issue view", "id", {**_GH, **_switches("--comments -c")}, _GH_EXCLUDED),
    Tool(
        "find",
        "find",
        {
            **_switches(
                "-P -empty -print -print0 -ls -prune -quit -a -and -o -or -not"
            ),
            "-maxdepth": "depth",
            "-mindepth": "depth",
            "-name": "basename",
            "-iname": "basename",
            "-path": "path",
            "-ipath": "path",
            "-type": "file_type",
            "-size": "size",
            "-mtime": "signed_count",
            "-mmin": "signed_count",
        },
        "-exec/-execdir/-ok/-okdir, -delete, -fprint/-fprint0/-fprintf/-fls, -files0-from, -L/-H/-follow",
    ),
    Tool(
        "git ls-files",
        "paths",
        {
            **_switches(
                "--cached -c --deleted -d --modified -m --others -o --ignored -i --stage -s --directory --no-empty-directory --unmerged -u --killed -k --exclude-standard --full-name --error-unmatch --deduplicate -z"
            ),
        },
        "--exclude-from, --recurse-submodules, --format",
    ),
    Tool("cat", "files", _switches("-n"), "unknown flags, shell syntax"),
    Tool("head", "files", {"-n": "count"}, "unknown flags, shell syntax"),
    Tool("tail", "files", {"-n": "count"}, "-f/--follow, unknown flags"),
    Tool("wc", "files", _switches("-l -w -m"), "unknown flags, shell syntax"),
    Tool("git show", "patch", _PATCH, _GIT_EXCLUDED),
    Tool("git diff", "patch", {**_PATCH, "--cached": None}, _GIT_EXCLUDED),
    Tool(
        "gh issue list",
        "none",
        {**_GH, "--limit": "limit", "--state": "state"},
        _GH_EXCLUDED,
    ),
    Tool(
        "gh pr view",
        "pr",
        {**_GH, "--json": "pr_fields", **_switches("--comments -c")},
        _GH_EXCLUDED,
    ),
)


def _option(name: str, kind: str | None, cluster: str) -> str:
    flag = _literal(name)
    if len(name) == 2 and name.startswith("-") and cluster:
        flag = "-[" + cluster + "]*" + name[1]
    if kind is None:
        return flag
    optional = kind.endswith("?")
    kind = kind.removesuffix("?")
    separator = "(?:" + _SPACE + "|=)" if name.startswith("--") else _SPACE
    alternatives = [flag + separator + _value(kind)]
    if len(name) == 2 and kind in ("count", "depth", "bytes"):
        alternatives.append(
            flag + {"count": _COUNT, "depth": _DEPTH, "bytes": _BYTES}[kind]
        )
    if optional:
        alternatives.append(flag)
    return "(?:" + "|".join(alternatives) + ")"


def _flags(options: dict[str, str | None], *, history: bool = False) -> str:
    short = "".join(
        name[1] for name, kind in options.items() if len(name) == 2 and kind is None
    )
    cluster = re.escape(short).replace(r"\-", "-")
    alternatives = []
    if short:
        alternatives.append("-[" + cluster + "]+")
    alternatives.extend(
        _option(name, kind, cluster)
        for name, kind in options.items()
        if len(name) != 2 or kind is not None
    )
    if history:
        alternatives.append("-" + _COUNT)
    return (
        "(?:" + _SPACE + "(?:" + "|".join(alternatives) + "))*" if alternatives else ""
    )


def _bare_paths(minimum: int = 0) -> str:
    return "(?:" + _SPACE + _path() + ")" + ("+" if minimum else "*")


def _tail(flags: str, *, kind: str = "path", minimum: int = 0) -> str:
    operand = _SPACE + _value(kind) + flags
    end = "(?:" + _SPACE + "--" + _bare_paths() + ")?"
    zero = "(?:" + operand + ")*" + end
    if minimum:
        return "(?:" + operand + zero + "|" + _SPACE + "--" + _bare_paths(1) + ")"
    return zero


def _prefix(name: str) -> str:
    words = name.split()
    if words[0] == "git":
        return "git(?:" + _SPACE + "--no-pager)?" + _SPACE + _SPACE.join(words[1:])
    return _SPACE.join(words)


def _forms(tool: Tool) -> list[str]:
    prefix = _prefix(tool.name)
    options = {
        name: kind for name, kind in tool.flags.items() if name not in ("--files", "-P")
    }
    flags = _flags(options, history=tool.shape == "history")
    start = prefix + flags
    shape = tool.shape
    if shape == "find":
        return [prefix + "(?:" + _SPACE + "-P)?" + _bare_paths() + flags]
    if shape in ("search", "search_file"):
        minimum = int(shape == "search_file")
        tail = _tail(flags, minimum=minimum)
        patterns = [_argument()]
        if "-e" in options:
            cluster = re.escape(
                "".join(
                    name[1]
                    for name, kind in options.items()
                    if len(name) == 2 and kind is None
                )
            )
            patterns.append(_option("-e", "argument", cluster))
        alternatives = [
            _SPACE + "(?:" + "|".join(patterns) + ")" + flags + tail,
            _SPACE + "--" + _SPACE + _argument() + _bare_paths(minimum),
        ]
        if tool.name == "rg":
            alternatives.append(_SPACE + "--files" + flags + _tail(flags))
        return [start + "(?:" + "|".join(alternatives) + ")"]
    if shape == "none":
        return [start]
    if shape in ("id", "pr"):
        value = (
            _value("issue_id")
            if shape == "id"
            else "(?:" + _value("pr_id") + "|" + _path() + ")"
        )
        return [start + _SPACE + value + flags]
    if shape == "patch":
        return [
            prefix
            + _SPACE
            + "--no-ext-diff"
            + _SPACE
            + "--no-textconv"
            + flags
            + _tail(flags, kind="history")
        ]
    forms = [
        start
        + _tail(
            flags,
            kind="history" if shape == "history" else "path",
            minimum=int(shape == "files"),
        )
    ]
    if tool.patch:
        patch_flags = _flags({**options, **tool.patch}, history=True)
        forms.append(
            prefix
            + _SPACE
            + "--no-ext-diff"
            + _SPACE
            + "--no-textconv"
            + patch_flags
            + _tail(patch_flags, kind="history")
        )
    return forms


def _deny_rule() -> str:
    unsafe = (
        r"^(?s:.*[\x00-\x1f\x7f"
        + "".join("\\" + c if c in "\\]-^" else c for c in _UNSAFE)
        + r"].*)$"
    )
    return "command(regex:" + unsafe + ")"


@cache
def _rules_by_tool() -> dict[str, tuple[str, ...]]:
    return {
        tool.name: tuple("command(regex:^" + form + r"\x20*$)" for form in _forms(tool))
        for tool in TOOLS
    }


def command_permissions() -> dict[str, list[str]]:
    """Unknown flags are never operands, even after a '--' delimiter."""
    return {
        "allow": [rule for rules in _rules_by_tool().values() for rule in rules],
        "deny": [_deny_rule()],
    }


@cache
def _compile(rule: str) -> re.Pattern[str]:
    return re.compile(rule[len("command(regex:") : -1])


def command_allowed(command: str) -> bool:
    """Use exactly the settings' deny-first rules on the unmodified command."""
    if _compile(_deny_rule()).search(command):
        return False
    # Hooks are fresh processes. Compile just the relevant tool's rules, rather
    # than recompiling all tools on every command. This dispatch can only
    # narrow candidates; the original anchored regex still decides permission.
    words = command.split(maxsplit=4)
    if len(words) > 1 and words[:2] == ["git", "--no-pager"]:
        words.pop(1)
    count = 3 if words[:1] == ["gh"] else 2 if words[:1] == ["git"] else 1
    rules = _rules_by_tool().get(" ".join(words[:count]), ())
    return any(_compile(rule).search(command) for rule in rules)


def permission_table() -> str:
    """Render the README table from the same definitions (checked by unit tests)."""
    rows = [
        "| 道具 | フラグ数（別名を含む） | 許可するフラグと値の型 | 明示的に外すもの |",
        "|---|---:|---|---|",
    ]
    for tool in TOOLS:
        flags = {**tool.flags, **tool.patch}
        rendered = ", ".join(
            "`" + name + (" " + kind if kind else "") + "`"
            for name, kind in flags.items()
        )
        rows.append(f"| `{tool.name}` | {len(flags)} | {rendered} | {tool.excluded} |")
    return "\n".join(rows)


def readonly_environment(environ: Mapping[str, str], *, cwd: Path) -> dict[str, str]:
    """Override repository helpers without modifying any user/repository config.

    Drop inherited Git path/config/exec overrides, including CONFIG_PARAMETERS
    (which would override CONFIG_COUNT). Keep gh credentials in the child only.
    """
    env = {key: value for key, value in environ.items() if not key.startswith("GIT_")}
    env.update(
        {
            "AGY_CLI_DISABLE_AUTO_UPDATE": "true",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GH_PAGER": "cat",
            "RIPGREP_CONFIG_PATH": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ATTR_NOSYSTEM": "1",
        }
    )
    overrides = {
        "core.fsmonitor": "false",
        "core.hooksPath": "/dev/null",
        "core.pager": "cat",
        "log.showSignature": "false",
        "format.pretty": "medium",
        "submodule.recurse": "false",
        "diff.ignoreSubmodules": "all",
    }
    env["GIT_CONFIG_COUNT"] = str(len(overrides))
    for index, (key, value) in enumerate(overrides.items()):
        env[f"GIT_CONFIG_KEY_{index}"] = key
        env[f"GIT_CONFIG_VALUE_{index}"] = value
    # --no-textconv does not disable attributes' clean/process filters. Git can
    # run them even while comparing working-tree files for a read-only diff.
    # Enumerate only names via Git's config reader (which executes no helpers),
    # including local/worktree include files, then override every such driver.
    try:
        result = subprocess.run(
            [
                "git",
                "config",
                "--null",
                "--name-only",
                "--get-regexp",
                r"^filter\..*\.(clean|smudge|process|required)$",
            ],
            cwd=cwd,
            env=env,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result.returncode not in (0, 1):  # 1 means no matching keys.
            raise ValueError("cannot inspect workspace Git filters")
        keys = result.stdout.decode("utf-8").split("\0")
        drivers = {key.rsplit(".", 1)[0] for key in keys if key}
        if any(
            any(ord(char) < 32 or ord(char) == 127 for char in name) for name in drivers
        ):
            raise ValueError("invalid workspace Git filter name")
        for name in sorted(drivers):
            for suffix, value in (
                ("clean", ""),
                ("smudge", ""),
                ("process", ""),
                ("required", "false"),
            ):
                index = len(overrides)
                key = name + "." + suffix
                overrides[key] = value
                env[f"GIT_CONFIG_KEY_{index}"] = key
                env[f"GIT_CONFIG_VALUE_{index}"] = value
        env["GIT_CONFIG_COUNT"] = str(len(overrides))
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise ValueError("cannot inspect workspace Git filters") from exc
    return env
