"""CPU-only fresh-process SHA diagnostics; no torch or model imports.

Run using .venv/bin/python and wrap in `timeout --kill-after=5s 590s` for
a hard envelope below 10 minutes. Fixed passes never retry until success.
The OpenSSL mask affects only copied child environments; it is a diagnostic
intervention, not a production fix, and may not affect coreutils sha256sum.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / ".venv/bin/python"
EXPECTED = "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc"
KNOWN_MEMORY_EXPECTED = (
    "281e519df3077b557c6b03f5da83c4e8d397219259615dd7c3308f89cae8f2a6"
)
CHECKPOINT = Path(
    "/home/kamimura/projects/tennis-lab/third_party/GVHMR/inputs/checkpoints/"
    "vitpose/vitpose-h-multi-coco.pth"
)
MASK = ":~0x20000000"
MEMORY_ITERATIONS = 4
CONDITIONS = ("inherited", "masked_sha_extension")


def child_environment(condition: str) -> dict[str, str]:
    environment = os.environ.copy()
    if condition == "masked_sha_extension":
        environment["OPENSSL_ia32cap"] = MASK
    elif condition != "inherited":
        raise ValueError(f"Unknown condition: {condition}")
    return environment


def environment_record(environment: dict[str, str]) -> dict[str, str | None]:
    # Do not expose unrelated credentials from the inherited environment.
    keys = (
        "OPENSSL_ia32cap",
        "OPENSSL_CONF",
        "OPENSSL_MODULES",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PATH",
    )
    return {key: environment.get(key) for key in keys}


def stat_record() -> dict[str, int]:
    stat = CHECKPOINT.stat()
    return {
        key: getattr(stat, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    }


def memory_child() -> int:
    # Imported only inside this fresh child, after its environment is installed.
    import _hashlib
    import _sha256
    import hashlib
    import ssl

    buffer = bytes(range(256)) * (64 * 1024 * 1024 // 256)
    rows = []
    for iteration in range(MEMORY_ITERATIONS):
        started = time.monotonic()
        openssl = hashlib.sha256(buffer).hexdigest()
        software = _sha256.sha256(buffer).hexdigest()
        rows.append(
            {
                "iteration": iteration,
                "hashlib": openssl,
                "_sha256": software,
                "expected": KNOWN_MEMORY_EXPECTED,
                "seconds": time.monotonic() - started,
                "match": openssl == software == KNOWN_MEMORY_EXPECTED,
            }
        )
    match = all(
        row["hashlib"] == row["_sha256"] == KNOWN_MEMORY_EXPECTED for row in rows
    )
    report = {
        "scope": "known_memory",
        "pid": os.getpid(),
        "python": sys.version,
        "executable": sys.executable,
        "openssl": ssl.OPENSSL_VERSION,
        "environment": environment_record(dict(os.environ)),
        "pattern": "bytes(range(256)) repeated to 64 MiB, one immutable buffer reused",
        "buffer_bytes": len(buffer),
        "expected": KNOWN_MEMORY_EXPECTED,
        "rows": rows,
        "match": match,
        "hashlib_type": str(type(hashlib.sha256())),
        "software_type": str(type(_sha256.sha256())),
        "module_origins": {
            module.__name__: getattr(module, "__file__", None)
            or getattr(getattr(module, "__spec__", None), "origin", None)
            for module in (_hashlib, _sha256, hashlib)
        },
    }
    print(json.dumps(report), flush=True)
    return 0 if match else 1


class Probe:
    def __init__(self, output: Path, iterations: int = 8) -> None:
        if not 1 <= iterations <= 8:
            raise ValueError("iterations must be between 1 and 8")
        output.mkdir(parents=True, exist_ok=True)
        self.output = output / "results.json"
        with self.output.open("x") as stream:
            stream.write('{"status":"initializing"}\n')
        self.started = time.monotonic()
        self.iterations = iterations
        self.data: dict[str, Any] = {
            "status": "running",
            "checkpoint": str(CHECKPOINT),
            "expected": EXPECTED,
            "iterations_per_condition": iterations,
            "memory_iterations": MEMORY_ITERATIONS,
            "file_rows": [],
            "memory_rows": [],
            "provenance_commands": [],
            "provenance": {
                "python": sys.version,
                "executable": sys.executable,
                "child_python": str(PYTHON),
                "platform": platform.platform(),
                "conditions": {
                    condition: environment_record(child_environment(condition))
                    for condition in CONDITIONS
                },
                "mask_source": "https://docs.openssl.org/3.5/man3/OPENSSL_ia32cap/",
                "mask": MASK,
                "interpretation": "Child-local diagnostic intervention, not a production fix. "
                "The mask targets OpenSSL SHA-extension dispatch; coreutils may use another "
                "implementation and ignore it. Inherited condition retains any existing mask. "
                "OpenSSL 3.5 documents that this two-vector syntax also zeroes later "
                "capability vectors, so the intervention is not restricted to the SHA bit. "
                "Alternating fresh processes do not eliminate chronological confounding.",
            },
        }
        self.save()

    def save(self) -> None:
        self.data["elapsed_seconds"] = time.monotonic() - self.started
        temporary = self.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n")
        temporary.replace(self.output)

    def command(
        self, arguments: list[str], condition: str, timeout: float
    ) -> dict[str, Any]:
        remaining = 560 - (time.monotonic() - self.started)
        if remaining <= 0:
            raise TimeoutError("Overall 560 second internal command budget exhausted")
        started = time.monotonic()
        row: dict[str, Any] = {
            "command": arguments,
            "condition": condition,
            "OPENSSL_ia32cap": child_environment(condition).get("OPENSSL_ia32cap"),
        }
        try:
            process = subprocess.run(
                arguments,
                env=child_environment(condition),
                capture_output=True,
                text=True,
                timeout=min(timeout, remaining),
                check=False,
            )
            row.update(
                exit_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:

            def decode(value: str | bytes | None) -> str:
                return (
                    value.decode(errors="replace")
                    if isinstance(value, bytes)
                    else value or ""
                )

            row.update(
                exit_code=None,
                stdout=decode(exc.stdout),
                stderr=decode(exc.stderr),
                timed_out=True,
            )
        except OSError as exc:
            row.update(exit_code=None, stdout="", stderr=repr(exc), timed_out=False)
        row["seconds"] = time.monotonic() - started
        return row

    def provenance(self) -> None:
        executable = shutil.which("sha256sum")
        if executable is None:
            raise FileNotFoundError("sha256sum is required")
        self.data["sha256sum_path"] = executable
        for command in (
            ["openssl", "version", "-a"],
            [executable, "--version"],
            ["ldd", executable],
        ):
            self.data["provenance_commands"].append(
                self.command(command, "inherited", 10)
            )
            self.save()

    def file_pass(self, condition: str, iteration: int) -> None:
        before = stat_record()
        row = self.command(
            [self.data["sha256sum_path"], str(CHECKPOINT)], condition, 30
        )
        after = stat_record()
        tokens = row["stdout"].split()
        digest = tokens[0] if tokens else None
        row.update(
            scope="file",
            iteration=iteration,
            digest=digest,
            expected=EXPECTED,
            stat_before=before,
            stat_after=after,
            match=row["exit_code"] == 0
            and digest == EXPECTED
            and before == after == self.data["initial_stat"],
        )
        self.data["file_rows"].append(row)
        self.save()
        print(json.dumps(row), flush=True)

    def memory_pass(self, condition: str) -> None:
        row = self.command(
            [str(PYTHON), "-B", str(Path(__file__).resolve()), "--memory-child"],
            condition,
            30,
        )
        try:
            report = json.loads(row["stdout"])
        except json.JSONDecodeError:
            report = None
        row.update(
            scope="known_memory",
            report=report,
            match=row["exit_code"] == 0
            and isinstance(report, dict)
            and report.get("match") is True,
        )
        self.data["memory_rows"].append(row)
        self.save()
        print(json.dumps(row), flush=True)

    def summarize(self) -> None:
        file_rows = self.data["file_rows"]
        memory_rows = self.data["memory_rows"]
        digests = {
            child_row[method]
            for row in memory_rows
            if isinstance(row["report"], dict)
            for child_row in row["report"]["rows"]
            for method in ("hashlib", "_sha256")
        }
        summary = {
            "file": {
                condition: {
                    "runs": sum(row["condition"] == condition for row in file_rows),
                    "mismatches_or_errors": sum(
                        row["condition"] == condition and not row["match"]
                        for row in file_rows
                    ),
                }
                for condition in CONDITIONS
            },
            "known_memory": {
                "expected": KNOWN_MEMORY_EXPECTED,
                "children": len(memory_rows),
                "all_children_match": all(row["match"] for row in memory_rows),
                "unique_digests_across_conditions_and_implementations": sorted(digests),
                "cross_condition_match": len(digests) == 1,
                "all_digests_match_expected": digests == {KNOWN_MEMORY_EXPECTED},
            },
        }
        self.data["summary"] = summary
        passed = (
            len(file_rows) == 2 * self.iterations
            and all(row["match"] for row in file_rows)
            and len(memory_rows) == 2
            and all(row["match"] for row in memory_rows)
            and digests == {KNOWN_MEMORY_EXPECTED}
        )
        provenance_failed = any(
            row["exit_code"] != 0 for row in self.data["provenance_commands"]
        )
        self.data["status"] = (
            "error" if provenance_failed else "passed" if passed else "mismatch"
        )
        self.save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--memory-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use the repository .venv/bin/python (Python 3.11)")
    if args.memory_child:
        return memory_child()
    if args.output_dir is None:
        parser.error("--output-dir is required")
    probe = Probe(args.output_dir, args.iterations)
    try:
        probe.data["initial_stat"] = stat_record()
        probe.provenance()
        for iteration in range(args.iterations):
            for condition in CONDITIONS:
                probe.file_pass(condition, iteration)
        for condition in CONDITIONS:
            probe.memory_pass(condition)
        probe.summarize()
    except BaseException:
        probe.data["status"] = "error"
        probe.data["error"] = traceback.format_exc()
        raise
    finally:
        probe.save()
    return 0 if probe.data["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
