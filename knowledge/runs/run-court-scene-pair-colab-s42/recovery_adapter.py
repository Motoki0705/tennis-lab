"""Versioned, closed-evaluation-only I/O adapter for the frozen Court campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

import campaign as c
import runtime as rt

VERSION = "court-closed-evaluation-recovery-v1"
SCIENTIFIC_REF = "17ad301bf8d80a5d094f28bb48f3431c7d9b5d89"
OBJECTS = {
    "src": "5c65b7156a169489f3fa45be7ca9f7b3f37ef60b",
    "uv.lock": "4ca6e7452b403d015fba963b6517d90373263951",
    "third_party/dinov3": "4add255125e6cbf446c6a43c5f3028aa41b63648",
}
FILES = {
    "campaign.py": "4ad82acddba97bf88d6a16e80d8d4ac6f5f88983295c63c6e75a48744b4ff615",
    "runtime.py": "abdf0a9dd2ed0860e8c5994398b5abba211837137b8fb2cc246503524f1e2b9e",
    "bootstrap.sh": "9a34cd12e640e2d25f666a0cf0e2e1cc386ee7b35877b292d32f8cf6238235af",
}


def protected_source(
    repo: Path, *, require_frozen_head: bool = True, require_submodule: bool = True
) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    head = git("rev-parse", "HEAD")
    if require_frozen_head and head != SCIENTIFIC_REF:
        raise ValueError("VM scientific HEAD changed")
    objects = {name: git("rev-parse", f"HEAD:{name}") for name in OBJECTS}
    if objects != OBJECTS:
        raise ValueError("Protected scientific Git objects changed")
    bundle = repo / "knowledge/runs/run-court-scene-pair-colab-s42"
    files = {name: c.sha(bundle / name) for name in FILES}
    if files != FILES:
        raise ValueError("Protected campaign/runtime/bootstrap bytes changed")
    paths = [*OBJECTS, *(str((bundle / name).relative_to(repo)) for name in FILES)]
    if git("status", "--porcelain", "--untracked-files=all", "--", *paths):
        raise ValueError("Protected scientific working tree changed")
    if require_submodule:
        observed = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo / "third_party/dinov3", text=True
        ).strip()
        if observed != OBJECTS["third_party/dinov3"]:
            raise ValueError("DINO checkout does not match scientific gitlink")
    return {
        "scientific_ref": SCIENTIFIC_REF,
        "checkout_head": head,
        "source_objects": objects,
        "bundle_sha256": files,
        "protected_source_changed": False,
    }


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.as_posix() != value
        or value == "."
    ):
        raise ValueError("Unsafe or noncanonical evaluation path")
    return path


def local_path(relative: Path) -> Path:
    path = cast(Path, c.REPO) / relative
    if not path.is_relative_to(c.REPO):
        raise ValueError("Local path escapes repository")
    for item in [path, *path.parents]:
        if item == c.REPO.parent:
            break
        if item.is_symlink():
            raise ValueError("Local evaluation symlink is forbidden")
    return path


class RecoveryBackend(rt.ColabBackend):
    """Retain original scientific validators; qualify only one closed eval tree."""

    def __init__(self, drive_cli: Path) -> None:
        super().__init__()
        if not drive_cli.is_absolute() or not drive_cli.is_file():
            raise ValueError("An exact versioned operational Drive CLI is required")
        self.drive_cli = drive_cli
        self._qualified: dict[Path, str] | None = None

    def cli(self, *arguments: str) -> dict[str, Any]:
        # This changes the explicitly versioned CLI location, not its checks.
        run = subprocess.run(
            [sys.executable, str(self.drive_cli), *arguments],
            cwd=c.REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        if run.returncode:
            raise RuntimeError(
                f"Operational Drive CLI failed ({run.returncode}): "
                f"{run.stdout[-2000:]} {run.stderr[-1000:]}"
            )
        return cast(dict[str, Any], json.loads(run.stdout))

    def event(self, value: dict[str, Any]) -> None:
        item = {"schema": VERSION, "at": datetime.now(UTC).isoformat(), **value}
        print(json.dumps(item, sort_keys=True), flush=True)
        directory = os.environ.get("COURT_RECOVERY_EVIDENCE_DIR")
        if directory:
            target = Path(directory) / "adapter-progress.json"
            c.write(target, item)

    def fetch(self, relative: Path, *, refresh: bool = False) -> Path:
        # Never intercept refresh, checkpoint, input, mutable or unqualified paths.
        if not refresh and self._qualified is not None and relative in self._qualified:
            path = local_path(relative)
            if not path.is_file() or c.sha(path) != self._qualified[relative]:
                raise ValueError("Qualified evaluation local bytes changed")
            return path
        return cast(Path, super().fetch(relative, refresh=refresh))

    def expected(
        self, stage: c.Stage, result: dict[str, Any]
    ) -> tuple[Path, dict[Path, str]]:
        assert stage.condition is not None
        root = c.eval_root(stage.condition)
        if result["evaluation"] != (root / "evaluation.json").as_posix():
            raise ValueError("Closed evaluation report path changed")
        files = {}
        for name, sha in result["evaluation_files"].items():
            path = Path(safe_relative(name))
            if (
                not path.is_relative_to(root)
                or path == root
                or not isinstance(sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", sha)
            ):
                raise ValueError(
                    "Evaluation artifact escapes its exact qualified stage"
                )
            files[path] = sha
        if root / "evaluation.json" not in files:
            raise ValueError("Closed evaluation inventory lacks its report")
        return root, files

    def snapshot(self, root: Path, files: dict[Path, str]) -> dict[str, Any]:
        member = self.drive.resolve(root.as_posix())  # original unique ancestry guard
        if not member or not member.get("IsDir") or not member.get("ID"):
            raise ValueError("Closed evaluation directory identity missing")
        values = self.drive.json(
            [
                "lsjson",
                self.drive.remote_path(root.as_posix()),
                "--recursive",
                "--hash",
                "--hash-type",
                "md5",
                "--no-mimetype",
            ]
        )
        if not isinstance(values, list):
            raise ValueError("Invalid closed evaluation inventory")
        expected_files = {p.relative_to(root).as_posix() for p in files}
        expected_dirs = {
            parent.as_posix()
            for name in expected_files
            for parent in PurePosixPath(name).parents
            if parent.as_posix() != "."
        }
        entries: dict[str, Any] = {}
        identifiers = {member["ID"]}
        for item in values:
            name = safe_relative(item["Path"]).as_posix()
            identity = item.get("ID")
            if name in entries or not identity or identity in identifiers:
                raise ValueError("Ambiguous evaluation subtree path or ID")
            identifiers.add(identity)
            is_dir = item.get("IsDir")
            if type(is_dir) is not bool:
                raise ValueError("Evaluation inventory type missing")
            size, digest = item.get("Size"), item.get("Hashes", {}).get("md5")
            if not is_dir and (
                type(size) is not int
                or size < 0
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{32}", digest)
            ):
                raise ValueError("Evaluation size/content hash missing")
            entries[name] = {
                "ID": identity,
                "IsDir": is_dir,
                "Size": size if not is_dir else None,
                "md5": digest if not is_dir else None,
                "ModTime": item.get("ModTime"),
            }
        actual_files = {n for n, v in entries.items() if not v["IsDir"]}
        actual_dirs = set(entries) - actual_files
        if actual_files != expected_files or actual_dirs != expected_dirs:
            raise ValueError("Closed evaluation membership missing or extra")
        return {"root_ID": member["ID"], "entries": entries}

    def check_local(
        self, target: Path, root: Path, files: dict[Path, str], snapshot: dict[str, Any]
    ) -> None:
        if target.is_symlink() or not target.is_dir():
            raise ValueError("Local evaluation root is missing or a symlink")
        actual_files: set[str] = set()
        actual_dirs: set[str] = set()
        for path in target.rglob("*"):
            if path.is_symlink():
                raise ValueError("Local evaluation symlink is forbidden")
            name = path.relative_to(target).as_posix()
            (actual_dirs if path.is_dir() else actual_files).add(name)
            if not path.is_dir() and not path.is_file():
                raise ValueError("Local evaluation is not a regular file")
        entries = snapshot["entries"]
        if actual_files != {
            n for n, v in entries.items() if not v["IsDir"]
        } or actual_dirs != {n for n, v in entries.items() if v["IsDir"]}:
            raise ValueError("Local evaluation membership missing or extra")
        for relative, sha in files.items():
            name = relative.relative_to(root).as_posix()
            path = target / name
            with path.open("rb") as stream:
                md5 = hashlib.file_digest(stream, "md5").hexdigest()
            if (
                path.stat().st_size != entries[name]["Size"]
                or md5 != entries[name]["md5"]
                or c.sha(path) != sha
            ):
                raise ValueError("Closed evaluation local SHA/size/MD5 changed")

    def verify(self, stage: c.Stage, result: dict[str, Any]) -> None:
        marker = self.state / "done" / f"{stage.key}.json"
        if stage.kind != "evaluate" or not marker.exists():
            super().verify(stage, result)
            return
        if self._qualified is not None or marker.is_symlink():
            raise ValueError("Nested qualification or completion-marker symlink")
        receipt = c.read(marker)
        identity = c.digest(self.ready())
        if (
            receipt.get("schema") != c.SCHEMA
            or receipt.get("inputs_identity") != identity
            or receipt.get("stage")
            != {"kind": stage.kind, "condition": stage.condition}
            or receipt.get("result") != result
        ):
            raise ValueError("Closed evaluation completion/input identity changed")
        marker_sha = c.sha(marker)
        root, files = self.expected(stage, result)
        local = local_path(root)
        before = self.snapshot(root, files)
        self.event(
            {
                "phase": "closed-evaluation-prefetch",
                "stage": stage.key,
                "file_count": len(files),
                "bytes": sum(
                    v["Size"] for v in before["entries"].values() if not v["IsDir"]
                ),
            }
        )
        if local.exists():
            # Do not silently repair a corrupt existing canonical directory.
            self.check_local(local, root, files, before)
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".court-eval-qualify-", dir=local.parent
            ) as staging:
                target = Path(staging) / "tree"
                transfer = self.cli(
                    "download", root.as_posix(), str(target), "--verify"
                )
                if transfer.get("verified") is not True:
                    raise ValueError("Bulk evaluation transfer was not verified")
                self.check_local(target, root, files, before)
                target.rename(local)
        try:
            self._qualified = files
            # Original lineage, config, report schema/count/order/metrics and every SHA.
            super().verify(stage, result)
            self.check_local(local, root, files, before)
            after = self.snapshot(root, files)
            if after != before:
                raise ValueError("Closed evaluation mutated during qualification")
            if c.sha(marker) != marker_sha or c.digest(self.ready()) != identity:
                raise ValueError(
                    "Completion/input identity mutated during qualification"
                )
            self.event(
                {
                    "phase": "closed-evaluation-qualified",
                    "stage": stage.key,
                    "file_count": len(files),
                    "remote_inventory_sha256": c.digest(after),
                    "completion_sha256": marker_sha,
                    "original_verify_passed": True,
                }
            )
        finally:
            self._qualified = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--drive-cli", type=Path, required=True)
    args = parser.parse_args()
    identity = protected_source(c.REPO)
    expected = os.environ["COURT_RECOVERY_ADAPTER_SHA256"]
    if c.sha(Path(__file__)) != expected:
        raise ValueError("Operational adapter bytes differ from deployment receipt")
    operational_ref = os.environ["COURT_RECOVERY_OPERATIONAL_REF"]
    if not re.fullmatch(r"[0-9a-f]{40}", operational_ref):
        raise ValueError("Exact operational commit is required")
    backend = RecoveryBackend(args.drive_cli)
    backend.event(
        {
            "phase": "operational-identity",
            **identity,
            "adapter_version": VERSION,
            "adapter_sha256": expected,
            "operational_ref": operational_ref,
            "Drive_CLI_sha256": c.sha(args.drive_cli),
        }
    )
    c.Campaign(c.REPO / c.STATE, backend).run()


if __name__ == "__main__":
    main()
