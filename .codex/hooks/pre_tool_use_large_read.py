#!/usr/bin/env python3
"""Deny Bash reads that could place too much evidence in the active context.

Codex command hooks cannot spawn built-in agents. This hook therefore only
blocks the pending tool call and tells the active Codex agent to delegate the
same evidence request to a fresh ``codebase_scout``.

Malformed hook input fails open. Once a valid Bash command is recognized as a
read candidate, however, an input whose output cannot be bounded safely is
denied. Set ``CODEX_LARGE_READ_THRESHOLD_BYTES`` to a positive integer to
override the default 20 KiB threshold.

This parser is a guardrail, not a complete Bash interpreter: arbitrary shell
programs can compute output in ways that cannot be predicted statically. It
recognizes common reads, searches, nested shells, and interpreter one-liners;
recognized read candidates fail closed when their terminal-visible output
cannot be bounded.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_THRESHOLD_BYTES = 20 * 1024
THRESHOLD_ENV = "CODEX_LARGE_READ_THRESHOLD_BYTES"
STATEMENT_TOKENS = {";", "&&", "||", "&", "\n", "(", ")"}
PIPE_TOKENS = {"|", "|&"}
REDIRECTIONS = {"<", ">", ">>", "<<", "<<<", "<>"}
READ_COMMANDS = {
    "awk",
    "cat",
    "egrep",
    "fgrep",
    "gawk",
    "grep",
    "head",
    "rg",
    "sed",
    "tail",
}
SED_RANGE = re.compile(r"\s*(\d+)\s*(?:,\s*(\d+)\s*)?p\s*;?\s*")
ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)
PYTHON_COMMAND = re.compile(r"python(?:\d+(?:\.\d+)*)?\Z")


@dataclass(frozen=True)
class ReadEstimate:
    label: str
    size: int | None


def _threshold_bytes() -> int:
    raw = os.environ.get(THRESHOLD_ENV)
    if raw is None:
        return DEFAULT_THRESHOLD_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_THRESHOLD_BYTES
    return value if value > 0 else DEFAULT_THRESHOLD_BYTES


def _shell_tokens(command: str) -> list[str] | None:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()\n")
        lexer.whitespace_split = True
        lexer.commenters = ""
        lexer.whitespace = " \t\r"
        return list(lexer)
    except ValueError:
        return None


def _pipelines(tokens: list[str]) -> list[list[list[str]]]:
    result: list[list[list[str]]] = []
    pipeline: list[list[str]] = []
    invocation: list[str] = []
    for token in tokens:
        if token in PIPE_TOKENS:
            if invocation:
                pipeline.append(invocation)
                invocation = []
            continue
        if token in STATEMENT_TOKENS:
            if invocation:
                pipeline.append(invocation)
                invocation = []
            if pipeline:
                result.append(pipeline)
                pipeline = []
            continue
        invocation.append(token)
    if invocation:
        pipeline.append(invocation)
    if pipeline:
        result.append(pipeline)
    return result


def _stdout_redirected(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if token not in {">", ">>", "&>", "&>>"}:
            continue
        descriptor = tokens[index - 1] if index > 0 and tokens[index - 1].isdigit() else "1"
        if descriptor != "2":
            return True
    return False


def _basename(token: str) -> str:
    return Path(token).name


def _command_index(tokens: list[str]) -> int | None:
    index = 0
    while index < len(tokens) and tokens[index] in {"!", "do", "else", "then", "{"}:
        index += 1
    while index < len(tokens) and ASSIGNMENT.fullmatch(tokens[index]):
        index += 1
    if index >= len(tokens):
        return None

    if _basename(tokens[index]) == "env":
        index += 1
        while index < len(tokens):
            token = tokens[index]
            if ASSIGNMENT.fullmatch(token):
                index += 1
                continue
            if token in {"-i", "--ignore-environment"}:
                index += 1
                continue
            break
    if index < len(tokens) and _basename(tokens[index]) == "command":
        index += 1
        while index < len(tokens) and tokens[index].startswith("-"):
            index += 1
    return index if index < len(tokens) else None


def _strip_redirections(args: list[str]) -> tuple[list[str], list[str]] | None:
    clean: list[str] = []
    input_files: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token.isdigit() and index + 1 < len(args) and args[index + 1] in REDIRECTIONS:
            index += 1
            token = args[index]
        if token in REDIRECTIONS:
            if index + 1 >= len(args):
                return None
            if token in {"<", "<>"}:
                input_files.append(args[index + 1])
            index += 2
            continue
        clean.append(token)
        index += 1
    return clean, input_files


def _resolve_file(cwd: Path, operand: str) -> Path | None:
    if operand == "-" or any(character in operand for character in "$`*?["):
        return None
    expanded = Path(os.path.expanduser(operand))
    path = expanded if expanded.is_absolute() else cwd / expanded
    return path.resolve(strict=False)


def _file_size(cwd: Path, operand: str) -> int | None:
    path = _resolve_file(cwd, operand)
    if path is None:
        return None
    try:
        metadata = path.stat()
    except OSError:
        return None
    if stat.S_ISREG(metadata.st_mode):
        return metadata.st_size
    if path == Path(os.devnull):
        return 0
    return None


def _total_file_size(cwd: Path, operands: list[str]) -> int | None:
    total = 0
    for operand in operands:
        size = _file_size(cwd, operand)
        if size is None:
            return None
        total += size
    return total


def _unknown_reason(label: str, threshold: int) -> str:
    return (
        f"{label} has no safely computable output upper bound "
        f"(large-read threshold: {threshold} bytes)."
    )


def _large_reason(label: str, size: int, threshold: int) -> str:
    return f"{label} may return {size} bytes (large-read threshold: {threshold} bytes)."


def _cat_estimate(
    args: list[str],
    cwd: Path,
    stdin_estimate: ReadEstimate | None,
) -> ReadEstimate:
    stripped = _strip_redirections(args)
    if stripped is None:
        return ReadEstimate("cat read", None)
    clean, input_files = stripped
    operands = list(input_files)
    options_done = False
    for token in clean:
        if not options_done and token == "--":
            options_done = True
        elif not options_done and token.startswith("-"):
            return ReadEstimate("cat with formatting options", None)
        else:
            operands.append(token)
    if not operands:
        if stdin_estimate is None:
            return ReadEstimate("cat reading stdin", None)
        return ReadEstimate("cat pipeline read", stdin_estimate.size)
    size = _total_file_size(cwd, operands)
    if size is None:
        return ReadEstimate("cat read", None)
    return ReadEstimate("cat read", size)


def _parse_sed(args: list[str]) -> tuple[bool, list[str], list[str]] | None:
    stripped = _strip_redirections(args)
    if stripped is None:
        return None
    clean, input_files = stripped
    quiet = False
    scripts: list[str] = []
    files: list[str] = []
    index = 0
    options_done = False
    while index < len(clean):
        token = clean[index]
        if not options_done and token == "--":
            options_done = True
            index += 1
            continue
        if not options_done and token in {"-n", "--quiet", "--silent"}:
            quiet = True
            index += 1
            continue
        if not options_done and token in {"-E", "-r", "-u", "-s"}:
            index += 1
            continue
        if not options_done and token in {"-e", "--expression"}:
            if index + 1 >= len(clean):
                return None
            scripts.append(clean[index + 1])
            index += 2
            continue
        if not options_done and token.startswith("--expression="):
            scripts.append(token.split("=", 1)[1])
            index += 1
            continue
        if not options_done and token.startswith("-e") and len(token) > 2:
            scripts.append(token[2:])
            index += 1
            continue
        if not options_done and token.startswith("-"):
            return None
        if not scripts:
            scripts.append(token)
        else:
            files.append(token)
        index += 1
    files.extend(input_files)
    return quiet, scripts, files


def _sed_ranges(scripts: list[str]) -> list[tuple[int, int]] | None:
    ranges: list[tuple[int, int]] = []
    for script in scripts:
        match = SED_RANGE.fullmatch(script)
        if match is None:
            return None
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        if start < 1 or end < start:
            return None
        ranges.append((start, end))
    return ranges


def _bounded_sed_bytes(path: Path, ranges: list[tuple[int, int]], limit: int) -> int | None:
    max_line = max(end for _, end in ranges)
    selected = 0
    scanned = 0
    line_number = 0
    try:
        with path.open("rb") as handle:
            while line_number < max_line:
                scan_remaining = limit - scanned
                if scan_remaining <= 0:
                    return None
                line = handle.readline(scan_remaining + 1)
                if not line:
                    break
                if len(line) > scan_remaining:
                    return None
                scanned += len(line)
                line_number += 1
                copies = sum(start <= line_number <= end for start, end in ranges)
                selected += len(line) * copies
                if selected >= limit:
                    return selected
    except OSError:
        return None
    return selected


def _sed_estimate(
    args: list[str],
    cwd: Path,
    limit: int,
    stdin_estimate: ReadEstimate | None,
) -> ReadEstimate:
    parsed = _parse_sed(args)
    if parsed is None:
        return ReadEstimate("sed read", None)
    quiet, scripts, operands = parsed
    if not operands:
        if stdin_estimate is None:
            return ReadEstimate("sed reading stdin", None)
        return ReadEstimate("sed pipeline read", None)
    sizes = _total_file_size(cwd, operands)
    if sizes is None:
        return ReadEstimate("sed read", None)
    if not quiet:
        return ReadEstimate("sed read", sizes)

    ranges = _sed_ranges(scripts)
    if ranges is None:
        return ReadEstimate("sed -n read", None)
    selected = 0
    for operand in operands:
        path = _resolve_file(cwd, operand)
        if path is None:
            return ReadEstimate("sed -n read", None)
        measured = _bounded_sed_bytes(path, ranges, limit - selected)
        if measured is None:
            return ReadEstimate("sed -n read", None)
        selected += measured
        if selected >= limit:
            break
    return ReadEstimate("sed -n read", selected)


def _parse_count(raw: str) -> int | None:
    if not raw.isdecimal():
        return None
    return int(raw)


def _parse_head_tail(
    args: list[str],
) -> tuple[str, int, list[str], bool] | None:
    stripped = _strip_redirections(args)
    if stripped is None:
        return None
    clean, input_files = stripped
    count_kind = "lines"
    count = 10
    files: list[str] = []
    verbose = False
    quiet = False
    options_done = False
    index = 0
    while index < len(clean):
        token = clean[index]
        if not options_done and token == "--":
            options_done = True
            index += 1
            continue
        if not options_done and token in {"-q", "--quiet", "--silent"}:
            quiet = True
            index += 1
            continue
        if not options_done and token in {"-v", "--verbose"}:
            verbose = True
            index += 1
            continue
        if not options_done and token in {"-n", "--lines", "-c", "--bytes"}:
            if index + 1 >= len(clean):
                return None
            parsed_count = _parse_count(clean[index + 1])
            if parsed_count is None:
                return None
            count_kind = "bytes" if token in {"-c", "--bytes"} else "lines"
            count = parsed_count
            index += 2
            continue
        if not options_done and token.startswith("--lines="):
            parsed_count = _parse_count(token.split("=", 1)[1])
            if parsed_count is None:
                return None
            count_kind = "lines"
            count = parsed_count
            index += 1
            continue
        if not options_done and token.startswith("--bytes="):
            parsed_count = _parse_count(token.split("=", 1)[1])
            if parsed_count is None:
                return None
            count_kind = "bytes"
            count = parsed_count
            index += 1
            continue
        if not options_done and re.fullmatch(r"-\d+", token):
            count_kind = "lines"
            count = int(token[1:])
            index += 1
            continue
        if not options_done and token.startswith("-"):
            return None
        files.append(token)
        index += 1
    files.extend(input_files)
    add_headers = verbose or (len(files) > 1 and not quiet)
    return count_kind, count, files, add_headers


def _head_line_bytes(path: Path, count: int, limit: int) -> int | None:
    selected = 0
    try:
        with path.open("rb") as handle:
            for _ in range(count):
                remaining = limit - selected
                if remaining <= 0:
                    return selected
                line = handle.readline(remaining + 1)
                if not line:
                    break
                selected += len(line)
                if selected >= limit:
                    return selected
    except OSError:
        return None
    return selected


def _tail_line_bytes(path: Path, count: int, limit: int) -> int | None:
    if count == 0:
        return 0
    try:
        size = path.stat().st_size
        read_size = min(size, limit)
        with path.open("rb") as handle:
            handle.seek(size - read_size)
            suffix = handle.read(read_size)
    except OSError:
        return None

    boundaries = [index for index, value in enumerate(suffix) if value == ord("\n")]
    if suffix.endswith(b"\n") and boundaries:
        boundaries.pop()
    if len(boundaries) >= count:
        return len(suffix) - boundaries[-count] - 1
    if size > read_size:
        return None
    return len(suffix)


def _head_tail_estimate(
    command_name: str,
    args: list[str],
    cwd: Path,
    limit: int,
    stdin_estimate: ReadEstimate | None,
) -> ReadEstimate:
    parsed = _parse_head_tail(args)
    if parsed is None:
        return ReadEstimate(f"{command_name} read", None)
    count_kind, count, operands, add_headers = parsed
    if not operands:
        if stdin_estimate is None:
            return ReadEstimate(f"{command_name} reading stdin", None)
        if count_kind == "bytes":
            size = None if stdin_estimate.size is None else min(stdin_estimate.size, count)
        else:
            size = stdin_estimate.size
        return ReadEstimate(f"{command_name} pipeline read", size)

    selected = 0
    for operand in operands:
        path = _resolve_file(cwd, operand)
        size = _file_size(cwd, operand)
        if path is None or size is None:
            return ReadEstimate(f"{command_name} read", None)
        if add_headers:
            selected += len(operand.encode("utf-8")) + 16
        remaining = limit - selected
        if remaining <= 0:
            return ReadEstimate(f"{command_name} read", selected)
        measured: int | None
        if count_kind == "bytes":
            measured = min(size, count)
        elif command_name == "head":
            measured = _head_line_bytes(path, count, remaining)
        else:
            measured = _tail_line_bytes(path, count, remaining)
        if measured is None:
            return ReadEstimate(f"{command_name} read", None)
        selected += measured
        if selected >= limit:
            break
    return ReadEstimate(f"{command_name} read", selected)


def _find_exec_estimate(tokens: list[str]) -> ReadEstimate | None:
    if not any(token in {"-exec", "-execdir"} for token in tokens):
        return None
    exec_index = next(
        index for index, token in enumerate(tokens) if token in {"-exec", "-execdir"}
    )
    for token in tokens[exec_index + 1 :]:
        if _basename(token) in READ_COMMANDS or re.search(
            r"(?:^|\s)(?:awk|cat|grep|head|rg|sed|tail)(?:\s|$)", token
        ):
            return ReadEstimate("find -exec file-content traversal", None)
    return None


def _git_estimate(args: list[str]) -> ReadEstimate | None:
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}:
            index += 2
            continue
        if token.startswith(("--git-dir=", "--work-tree=", "--namespace=")):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        break
    if index >= len(args) or args[index] not in {"diff", "show"}:
        return None
    subcommand = args[index]
    options = args[index + 1 :]
    if "--quiet" in options:
        return None

    patch_state: bool | None = None
    for option in options:
        if option in {"--no-patch", "-s"}:
            patch_state = False
        elif option in {"--patch", "-p", "-u"} or option.startswith(
            ("--patch-with-", "--unified=")
        ):
            patch_state = True
    if patch_state is True:
        return ReadEstimate(f"unbounded git {subcommand} patch", None)
    if patch_state is False:
        return None

    summary_prefixes = (
        "--stat",
        "--shortstat",
        "--numstat",
        "--name-only",
        "--name-status",
        "--summary",
        "--check",
    )
    if any(token.startswith(summary_prefixes) for token in options):
        return None
    return ReadEstimate(f"unbounded git {subcommand}", None)


def _search_estimate(
    command_name: str,
    args: list[str],
    cwd: Path,
    stdin_estimate: ReadEstimate | None,
) -> ReadEstimate | None:
    if any(token in {"--help", "--version", "-V"} for token in args):
        return None
    explicit_files: list[tuple[str, int]] = []
    has_directory = False
    for token in args:
        if token.startswith("-") or token == "":
            continue
        path = _resolve_file(cwd, token)
        if path is None:
            continue
        try:
            metadata = path.stat()
        except OSError:
            continue
        if stat.S_ISDIR(metadata.st_mode):
            has_directory = True
        elif stat.S_ISREG(metadata.st_mode):
            explicit_files.append((token, metadata.st_size))

    label = f"{command_name} search read"
    if has_directory:
        return ReadEstimate(label, None)
    if not explicit_files:
        if stdin_estimate is None:
            return ReadEstimate(label, None)
        return ReadEstimate(label, stdin_estimate.size)

    expanding_options = {
        "--byte-offset",
        "--color",
        "--column",
        "--json",
        "--line-number",
        "--only-matching",
        "-b",
        "-n",
        "-o",
    }
    if any(
        token in expanding_options
        or token.startswith(("--color=", "--heading", "--replace="))
        for token in args
    ):
        return ReadEstimate(label, None)

    if len(explicit_files) == 1 or "--no-filename" in args or "-N" in args:
        return ReadEstimate(label, sum(size for _, size in explicit_files))
    size = sum(
        source_size * (len(filename.encode("utf-8")) + 3)
        for filename, source_size in explicit_files
    )
    return ReadEstimate(label, size)


def _awk_estimate(
    args: list[str],
    cwd: Path,
    stdin_estimate: ReadEstimate | None,
) -> ReadEstimate:
    if any(token in {"--help", "--version", "-V"} for token in args):
        return ReadEstimate("awk metadata", 0)
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"-F", "-v"}:
            index += 2
            continue
        if token.startswith(("-F", "-v")):
            index += 1
            continue
        if token.startswith("-"):
            return ReadEstimate("awk read", None)
        break
    if index >= len(args):
        return ReadEstimate("awk reading stdin", None)
    program = re.sub(r"\s+", "", args[index])
    operands = [token for token in args[index + 1 :] if not ASSIGNMENT.fullmatch(token)]
    if operands:
        size = _total_file_size(cwd, operands)
    elif stdin_estimate is not None:
        size = stdin_estimate.size
    else:
        size = None
    if program not in {"1", "{print}", "{print$0}"}:
        return ReadEstimate("awk read with unbounded program output", None)
    if size is None:
        return ReadEstimate("awk pass-through read", None)
    return ReadEstimate("awk pass-through read", size + len(operands))


def _literal_read_path(call: ast.Call) -> str | None:
    function = call.func
    if not isinstance(function, ast.Attribute):
        return None
    if function.attr in {"read", "readlines"} and isinstance(function.value, ast.Call):
        opener = function.value
        if isinstance(opener.func, ast.Name) and opener.func.id == "open" and opener.args:
            first = opener.args[0]
            return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None
    if function.attr in {"read_bytes", "read_text"} and isinstance(function.value, ast.Call):
        constructor = function.value
        if (
            isinstance(constructor.func, ast.Name)
            and constructor.func.id == "Path"
            and constructor.args
        ):
            first = constructor.args[0]
            return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None
    return None


def _is_file_read_call(call: ast.Call) -> bool:
    function = call.func
    if not isinstance(function, ast.Attribute):
        return False
    if function.attr in {"read", "readlines"} and isinstance(function.value, ast.Call):
        opener = function.value.func
        return isinstance(opener, ast.Name) and opener.id == "open"
    return function.attr in {"read_bytes", "read_text"} and isinstance(
        function.value, ast.Call
    )


def _direct_python_read(node: ast.AST) -> tuple[str | None, bool]:
    if isinstance(node, ast.Call) and _is_file_read_call(node):
        return _literal_read_path(node), True
    return None, False


def _python_estimate(args: list[str], cwd: Path) -> ReadEstimate | None:
    try:
        code_index = args.index("-c") + 1
        code = args[code_index]
    except (ValueError, IndexError):
        return None
    if "read" not in code:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        if re.search(r"(?:print|stdout).*?(?:open|Path).*?read", code, re.DOTALL):
            return ReadEstimate("Python interpreter file read", None)
        return None

    assigned_reads: dict[str, str | None] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        path, is_read = _direct_python_read(value)
        if not is_read:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                assigned_reads[target.id] = path

    paths: list[str] = []
    unknown = False
    visible_read = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_print = isinstance(node.func, ast.Name) and node.func.id == "print"
        is_stdout_write = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "write"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "stdout"
        )
        if not (is_print or is_stdout_write):
            continue
        for argument in node.args:
            path, is_read = _direct_python_read(argument)
            if is_read:
                visible_read = True
                if path is None:
                    unknown = True
                else:
                    paths.append(path)
                continue
            if isinstance(argument, ast.Name) and argument.id in assigned_reads:
                visible_read = True
                assigned_path = assigned_reads[argument.id]
                if assigned_path is None:
                    unknown = True
                else:
                    paths.append(assigned_path)
                continue
            if any(
                (isinstance(child, ast.Call) and _is_file_read_call(child))
                or (isinstance(child, ast.Name) and child.id in assigned_reads)
                for child in ast.walk(argument)
            ):
                visible_read = True
                unknown = True
    if not visible_read:
        return None
    if unknown:
        return ReadEstimate("Python interpreter file read", None)
    size = _total_file_size(cwd, paths)
    return ReadEstimate(
        "Python interpreter file read",
        None if size is None else size + 1,
    )


def _nested_shell_code(args: list[str]) -> str | None:
    for index, token in enumerate(args):
        if token == "-c" or (token.startswith("-") and "c" in token[1:]):
            return args[index + 1] if index + 1 < len(args) else None
    return None


def _xargs_command(args: list[str]) -> list[str]:
    options_with_value = {
        "--arg-file",
        "--delimiter",
        "--eof",
        "--max-args",
        "--max-chars",
        "--max-lines",
        "--max-procs",
        "--process-slot-var",
        "--replace",
        "-E",
        "-I",
        "-L",
        "-P",
        "-a",
        "-d",
        "-n",
        "-s",
    }
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            return args[index + 1 :]
        if token in options_with_value:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return args[index:]
    return []


def _xargs_estimate(
    args: list[str],
    cwd: Path,
    limit: int,
    depth: int,
) -> ReadEstimate | None:
    command = _xargs_command(args)
    if not command:
        return None
    command_index = _command_index(command)
    if command_index is None:
        return None
    command_name = _basename(command[command_index])
    command_args = command[command_index + 1 :]

    if command_name in READ_COMMANDS:
        return ReadEstimate(f"xargs {command_name} file-content traversal", None)
    if command_name in {"bash", "dash", "sh", "zsh"}:
        nested = _nested_shell_code(command_args)
        if nested is None:
            return None
        estimate = _estimate_command(nested, cwd, limit, depth + 1)
        if estimate is None:
            return None
        return ReadEstimate(f"xargs nested shell {estimate.label}", None)
    if PYTHON_COMMAND.fullmatch(command_name):
        estimate = _python_estimate(command_args, cwd)
        if estimate is None:
            return None
        return ReadEstimate(f"xargs interpreter {estimate.label}", None)
    if command_name in {"node", "perl", "php", "ruby"}:
        return ReadEstimate(f"xargs {command_name} interpreter output", None)
    return None


def _invocation_estimate(
    tokens: list[str],
    cwd: Path,
    limit: int,
    stdin_estimate: ReadEstimate | None,
    depth: int,
) -> ReadEstimate | None:
    if depth > 4:
        return ReadEstimate("deeply nested read command", None)
    command_index = _command_index(tokens)
    if command_index is None:
        return None
    command_name = _basename(tokens[command_index])
    args = tokens[command_index + 1 :]
    if command_name == "find":
        return _find_exec_estimate(args)
    if command_name == "xargs":
        return _xargs_estimate(args, cwd, limit, depth)
    if command_name == "cat":
        return _cat_estimate(args, cwd, stdin_estimate)
    if command_name == "sed":
        return _sed_estimate(args, cwd, limit, stdin_estimate)
    if command_name in {"head", "tail"}:
        return _head_tail_estimate(
            command_name,
            args,
            cwd,
            limit,
            stdin_estimate,
        )
    if command_name == "git":
        return _git_estimate(args)
    if command_name in {"rg", "grep", "egrep", "fgrep"}:
        return _search_estimate(command_name, args, cwd, stdin_estimate)
    if command_name in {"awk", "gawk"}:
        return _awk_estimate(args, cwd, stdin_estimate)
    if PYTHON_COMMAND.fullmatch(command_name):
        return _python_estimate(args, cwd)
    if command_name in {"bash", "dash", "sh", "zsh"}:
        nested = _nested_shell_code(args)
        if nested is not None:
            return _estimate_command(nested, cwd, limit, depth + 1)
    return None


def _pipeline_estimate(
    pipeline: list[list[str]],
    cwd: Path,
    limit: int,
    depth: int,
) -> ReadEstimate | None:
    stdin_estimate: ReadEstimate | None = None
    for index, invocation in enumerate(pipeline):
        is_last = index == len(pipeline) - 1
        if _stdout_redirected(invocation):
            stdin_estimate = None
            if is_last:
                return None
            continue

        command_index = _command_index(invocation)
        command_name = (
            _basename(invocation[command_index]) if command_index is not None else ""
        )
        if stdin_estimate is not None and command_name == "wc":
            stdin_estimate = ReadEstimate("wc summary of Bash read", 128)
            continue

        estimate = _invocation_estimate(
            invocation,
            cwd,
            limit,
            stdin_estimate,
            depth,
        )
        if estimate is not None:
            stdin_estimate = estimate
        elif stdin_estimate is not None:
            stdin_estimate = ReadEstimate(
                f"pipeline output after {stdin_estimate.label}",
                None,
            )
    return stdin_estimate


def _estimate_command(
    command: str,
    cwd: Path,
    limit: int,
    depth: int = 0,
) -> ReadEstimate | None:
    if depth > 4:
        return ReadEstimate("deeply nested shell read", None)
    tokens = _shell_tokens(command)
    if tokens is None:
        return None
    total = 0
    labels: list[str] = []
    for pipeline in _pipelines(tokens):
        estimate = _pipeline_estimate(pipeline, cwd, limit - total, depth)
        if estimate is None:
            continue
        labels.append(estimate.label)
        if estimate.size is None:
            return ReadEstimate(" + ".join(labels), None)
        total += estimate.size
        if total >= limit:
            break
    if not labels:
        return None
    return ReadEstimate(" + ".join(labels), total)


def _assess_command(command: str, cwd: Path, threshold: int) -> str | None:
    estimate = _estimate_command(command, cwd, threshold)
    if estimate is None:
        return None
    if estimate.size is None:
        return _unknown_reason(estimate.label, threshold)
    if estimate.size >= threshold:
        return _large_reason(estimate.label, estimate.size, threshold)
    return None


def _deny_payload(reason: str, command: str) -> dict[str, object]:
    request = command if len(command) <= 500 else f"{command[:497]}..."
    instruction = (
        "Do not retry this large read in the root thread. Delegate the same evidence "
        f"request ({request!r}) with spawn_agent using agent_type=\"codebase_scout\" "
        "and fork_turns=\"none\". Ask for a terminal-only summary, then wait for that "
        "summary before continuing. This command hook cannot spawn built-in agents; "
        "the active Codex agent must perform the delegation."
    )
    return {
        "systemMessage": f"Large Bash read denied before execution: {reason}",
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
            "additionalContext": instruction,
        },
    }


def main() -> int:
    try:
        raw_payload = json.load(sys.stdin)
        if not isinstance(raw_payload, dict):
            return 0
        if raw_payload.get("hook_event_name") != "PreToolUse":
            return 0
        if raw_payload.get("tool_name") != "Bash":
            return 0
        tool_input = raw_payload.get("tool_input")
        if not isinstance(tool_input, dict):
            return 0
        command = tool_input.get("command", tool_input.get("cmd"))
        cwd_value = raw_payload.get("cwd")
        if not isinstance(command, str) or not isinstance(cwd_value, str):
            return 0
        cwd = Path(cwd_value)
        if not cwd.is_dir():
            return 0
        reason = _assess_command(command, cwd, _threshold_bytes())
        if reason is None:
            return 0
        print(json.dumps(_deny_payload(reason, command), ensure_ascii=False))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
