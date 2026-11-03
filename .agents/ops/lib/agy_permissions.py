"""Generate agy command permissions from a small, explicit argv grammar."""

from __future__ import annotations

import re

# Keep the grammar compatible with regular-expression engines without lookaround.
# No whitespace in the rule itself: spaces in a command are represented as \x20.
_SPACE = r"\x20+"
_UNSAFE = ";&|<>$`(){}\\"


def _excluding(characters: str) -> str:
    escaped = "".join("\\" + c if c in "\\]-^" else c for c in characters)
    return r"[^\x00-\x20\x7f" + escaped + "]"


def _argument() -> str:
    # Bare globs/tilde could expand to new options. Only quoted globs are allowed.
    bare = _excluding(_UNSAFE + "'\"*?[]~-") + _excluding(_UNSAFE + "'\"*?[]") + "*"
    # Use an explicit space alternative inside quotes; all other controls and
    # shell metacharacters remain forbidden, even when a shell would quote them.
    quoted = (
        _excluding(_UNSAFE + "'\"-") + "(?:" + _excluding(_UNSAFE + "'\"") + r"|\x20)*"
    )
    return "(?:" + bare + "|'" + quoted + "'|\"" + quoted + '")'


def _literal(value: str) -> str:
    return re.escape(value).replace(r"\-", "-")


def _value(kind: str) -> str:
    if kind == "argument":
        return _argument()
    choices = {
        "number": r"[0-9]+",
        "fields": r"[A-Za-z][A-Za-z0-9_]*(?:,[A-Za-z][A-Za-z0-9_]*)*",
        "repository": r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
        "state": r"(?:open|closed|all)",
    }
    raw = choices[kind]
    return "(?:" + raw + "|'" + raw + "'|\"" + raw + '")'


def _option(name: str, kind: str | None) -> str:
    flag = _literal(name)
    if kind is None:
        return flag
    value = _value(kind)
    separator = "(?:" + _SPACE + "|=)" if name.startswith("--") else _SPACE
    alternatives = [flag + separator + value]
    if len(name) == 2 and kind == "number":
        alternatives.append(flag + r"[0-9]+")
    return "(?:" + "|".join(alternatives) + ")"


def _form(
    command: str,
    options: dict[str, str | None],
    minimum: int,
    maximum: int | None = None,
) -> str:
    words = command.split(" ")
    prefix = _SPACE.join(_literal(word) for word in words)
    if words[0] == "git":
        prefix = (
            "git(?:"
            + _SPACE
            + "--no-pager)?"
            + _SPACE
            + _SPACE.join(_literal(word) for word in words[1:])
        )
    flag = "(?:" + "|".join(_option(name, kind) for name, kind in options.items()) + ")"
    flags = "(?:" + _SPACE + flag + ")*" if options else ""
    operand = _SPACE + _argument() + flags
    count = (
        "{" + str(minimum) + "," + (str(maximum) if maximum is not None else "") + "}"
    )
    return "^" + prefix + flags + "(?:" + operand + ")" + count + r"\x20*$"


def command_permissions() -> dict[str, list[str]]:
    """Return deny-first command rules; unknown flags never become operands."""
    search: dict[str, str | None] = dict.fromkeys(("-n", "-i", "-l", "-F", "-w"))
    globs: dict[str, str | None] = {
        "-g": "argument",
        "--glob": "argument",
        "--hidden": None,
    }
    history: dict[str, str | None] = {
        "--oneline": None,
        "-n": "number",
        "--follow": None,
        "--": None,
    }
    patch: dict[str, str | None] = {
        "-p": None,
        "--stat": None,
        "--name-only": None,
        "--name-status": None,
        "--": None,
    }
    gh: dict[str, str | None] = {
        "--json": "fields",
        "--repo": "repository",
        "-R": "repository",
    }
    forms: list[tuple[str, dict[str, str | None], int, int | None]] = [
        ("cat", {"-n": None, "--": None}, 1, None),
        ("ls", dict.fromkeys(("-a", "-l", "-d", "-h", "-1", "--")), 0, None),
        ("head", {"-n": "number", "--": None}, 1, None),
        ("tail", {"-n": "number", "--": None}, 1, None),
        ("wc", dict.fromkeys(("-l", "-w", "-m", "--")), 1, None),
        ("grep", search, 2, None),
        (
            "rg",
            {
                **search,
                **globs,
                "-C": "number",
                "-A": "number",
                "-B": "number",
                "--": None,
            },
            1,
            None,
        ),
        ("rg --files", globs, 0, None),
        # Patch-producing Git commands must suppress external diff/textconv,
        # including helpers selected by repository attributes/configuration.
        ("git log", history, 0, None),
        (
            "git log --no-ext-diff --no-textconv",
            {**history, "-p": None, "--stat": None},
            0,
            None,
        ),
        ("git show --no-ext-diff --no-textconv", patch, 0, None),
        ("git diff --no-ext-diff --no-textconv", {**patch, "--cached": None}, 0, None),
        ("git grep", {**search, "--no-textconv": None, "--": None}, 1, None),
        ("gh issue view", {**gh, "--comments": None}, 1, 1),
        ("gh issue list", {**gh, "--limit": "number", "--state": "state"}, 0, 0),
        ("gh pr view", {**gh, "--comments": None}, 1, 1),
    ]
    # Python's $ can match before a final LF under re.search. The explicit deny
    # also rejects that boundary, tabs and controls before any allow is considered.
    unsafe = (
        r"^(?s:.*[\x00-\x1f\x7f"
        + "".join("\\" + c if c in "\\]-^" else c for c in _UNSAFE)
        + r"].*)$"
    )
    return {
        "allow": ["command(regex:" + _form(*form) + ")" for form in forms],
        "deny": ["command(regex:" + unsafe + ")"],
    }
