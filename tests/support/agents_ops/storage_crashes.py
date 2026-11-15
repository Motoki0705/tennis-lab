"""Crash a real storage process at recorded filesystem mutations, without finally."""

from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import re
import stat
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.support.agents_ops.storage_tier import Lab, make_lab

SCENARIOS = (
    "mirror-new",
    "mirror-update",
    "mirror-hardlinks",
    "hydrate",
    "evict",
    "evict-hardlinks",
    "discard-mirror",
    "discard-hydrate",
    "partial-mirror",
    "partial-hydrate",
    "index-legacy",
    "checkpoint-new",
    "checkpoint-mirror",
    "checkpoint-hardlinks",
    "checkpoint-hydrate",
    "recover-publication",
    "recover-hydrate",
    "recover-complete-stage",
    "recover-evict",
    "recover-retire",
    "recover-discard",
    "orphan-recovery",
    "recover-copy",
    "recover-checkpoint",
    "recover-checkpoint-hydrate",
)


def prepare(scenario: str, root: Path) -> tuple[Lab, Any, Callable[[], Any]]:
    injected_quiet = None
    injected_copy = None
    if scenario.startswith("recover-"):
        original_scenario, match = {
            "recover-publication": (
                "mirror-update",
                lambda v: v.startswith("rename:") and "->E/data/task/v1" in v,
            ),
            "recover-hydrate": (
                "hydrate",
                lambda v: v.startswith("rename:") and "->D/data/task/v1" in v,
            ),
            "recover-complete-stage": (
                "mirror-update",
                lambda v: v == "json:publications:complete",
            ),
            "recover-evict": ("evict", lambda v: v == "append:removals:claimed:a.bin"),
            "recover-retire": (
                "mirror-update",
                lambda v: v == "append:removals:claimed:a.bin",
            ),
            "recover-discard": (
                "discard-mirror",
                lambda v: v == "append:removals:claimed:a.bin",
            ),
            "recover-copy": (
                "mirror-new",
                lambda v: (
                    v.startswith("create:E/data/task/.storage-tier-stage-")
                    and v.endswith("/a.bin")
                ),
            ),
            "recover-checkpoint": (
                "checkpoint-mirror",
                lambda v: v.startswith("rename:") and "->E/ckpt/README.md" in v,
            ),
            "recover-checkpoint-hydrate": (
                "checkpoint-hydrate",
                lambda v: v.startswith("rename:") and "->D/ckpt/README.md" in v,
            ),
        }[scenario]
        lab, unit, action = prepare(original_scenario, root)
        code, trace = run_child(
            lab, action, root / "setup-trace.jsonl", stop_when=match
        )
        assert code == 91, (scenario, code, trace)
        return lab, unit, lambda: lab.engine.run("recover", [], dry_run=False)
    lab = make_lab(root)
    if scenario == "index-legacy":
        local = lab.config.repo / "ckpt"
        local.mkdir()
        directory = lab.config.mirror / "ckpt"
        directory.mkdir()
        (directory / "README.md").write_bytes(b"legacy checkpoint documentation")
        unit = lab.layout.Unit("ckpt", "checkpoint-files")
        tree = lab.filesystem.snapshot(directory, hashes=True)
        lab.engine.record(
            lab.layout.Unit("ckpt", "checkpoint"), None, tree, dry_run=False
        )
        command = "index"
    elif scenario.startswith("checkpoint-"):
        directory = lab.config.repo / "ckpt"
        directory.mkdir()
        (directory / "README.md").write_bytes(b"old checkpoint documentation")
        (directory / "obsolete.txt").write_bytes(b"obsolete checkpoint file")
        unit = lab.layout.Unit("ckpt", "checkpoint-files")
        if scenario != "checkpoint-new":
            lab.mirror(unit)
        if scenario == "checkpoint-new":
            command = "mirror"
        elif scenario in {"checkpoint-mirror", "checkpoint-hardlinks"}:
            (directory / "README.md").write_bytes(b"new checkpoint documentation")
            if scenario == "checkpoint-hardlinks":
                os.link(directory / "README.md", directory / "extra.txt")
            else:
                (directory / "extra.txt").write_bytes(b"new checkpoint file")
            (directory / "obsolete.txt").unlink()
            command = "mirror"
        else:
            for name in ("README.md", "obsolete.txt"):
                (directory / name).unlink()
            command = "hydrate"
    else:
        unit = lab.unit()
        if scenario.endswith("hardlinks"):
            os.link(
                lab.config.repo / unit.path / "a.bin",
                lab.config.repo / unit.path / "nested/link.bin",
            )
        command = "mirror"
        if scenario != "mirror-new":
            lab.mirror(unit)
        if scenario in {"hydrate", "discard-hydrate", "partial-hydrate"}:
            assert lab.run("evict", unit)[0] == 0
            command = "hydrate"
        elif scenario == "orphan-recovery":
            lab.age(unit, 1)
            lab.orphan = (lab.config.mirror / unit.path).with_name(
                ".storage-tier-stage-legacy-orphan"
            )
            lab.orphan.mkdir()
            (lab.orphan / "unverified.bin").write_bytes(b"unrecorded orphan payload")
            command = "recover"
        elif scenario in {"evict", "evict-hardlinks"}:
            command = "evict"
        else:
            (lab.config.repo / unit.path / "a.bin").write_bytes(
                b"new verified output\n"
            )
            lab.age(unit, 1)
        if scenario.startswith("discard-"):
            original = lab.engine.quiet
            failed = False

            def quiet(
                unit: Any, path: Path, tree: Any, *, destructive: bool = False
            ) -> None:
                nonlocal failed
                prefix = (
                    ".storage-tier-hydrate-"
                    if command == "hydrate"
                    else ".storage-tier-stage-"
                )
                parent = (
                    lab.config.repo if command == "hydrate" else lab.config.mirror
                ) / unit.path
                if (
                    not destructive
                    and not failed
                    and list(parent.parent.glob(prefix + "*"))
                ):
                    failed = True
                    raise OSError("injected failure after verified copy")
                original(unit, path, tree, destructive=destructive)

            injected_quiet = quiet
        if scenario.startswith("partial-"):
            original_copy = lab.module.copy_tree

            def partial(
                source: Path, target: Path, expected: Any, **kwargs: Any
            ) -> None:
                original_copy(source, target, expected, **kwargs)
                (target / "unverified.bin").write_bytes(
                    b"unverified stage payload must survive"
                )
                raise OSError("injected unverified partial copy")

            injected_copy = partial
    expected_base = (
        lab.config.mirror if command in {"hydrate", "index"} else lab.config.repo
    )
    expected_root = expected_base / unit.path
    lab.wanted = {
        str(path.relative_to(expected_root)): lab.filesystem.read_bytes(path)
        for path in expected_root.rglob("*")
        if path.is_file()
    }
    lab.filesystem.write_json(
        lab.config.state / "active.json",
        {
            "schema_version": 1,
            "repository": str(lab.config.repo),
            "mirror_mount": str(lab.config.mount),
        },
    )

    def execute() -> Any:
        if injected_quiet is not None:
            lab.engine.quiet = injected_quiet
        if injected_copy is not None:
            lab.module.copy_tree = injected_copy
        if command == "recover":
            return lab.engine.run("recover", [], dry_run=False)
        return lab.run(command, unit)

    return lab, unit, execute


class Trace:
    def __init__(
        self,
        lab: Lab,
        output: Path,
        stop: int | None,
        stop_when: Callable[[str], bool] | None,
    ) -> None:
        self.lab, self.stop, self.stop_when = lab, stop, stop_when
        self.fd = os.open(output, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        self.count = 0
        self.original_write, self.original_fsync = os.write, os.fsync

    def label_path(self, path: Any, directory_fd: int | None = None) -> str | None:
        path = Path(os.fsdecode(path))
        if directory_fd is not None and not path.is_absolute():
            path = Path(os.readlink(f"/proc/self/fd/{directory_fd}")) / path
        for name, base in (("E", self.lab.config.mirror), ("D", self.lab.config.repo)):
            if path.is_relative_to(base):
                return re.sub(
                    r"[0-9a-f]{32,64}", "{id}", name + "/" + str(path.relative_to(base))
                )
        return None

    def emit(self, label: str | None) -> None:
        if label is None:
            return
        self.original_write(self.fd, (json.dumps(label) + "\n").encode())
        self.original_fsync(self.fd)
        if self.count == self.stop or (
            self.stop_when is not None and self.stop_when(label)
        ):
            os._exit(91)
        self.count += 1

    def install(self) -> None:
        original_mkdir, original_open = Path.mkdir, Path.open
        original_replace, original_unlink, original_rmdir = (
            os.replace,
            os.unlink,
            os.rmdir,
        )
        original_fsync = os.fsync
        original_chmod, original_utime, original_link = os.chmod, os.utime, os.link
        fs = self.lab.filesystem
        original_rename, original_claim, original_append = (
            fs.rename,
            fs.rename_at,
            fs.append_json,
        )

        def mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
            existed = os.path.lexists(path)
            original_mkdir(path, *args, **kwargs)
            if not existed:
                value = self.label_path(path)
                self.emit("mkdir:" + value if value else None)

        def opened(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            stream = original_open(path, mode, *args, **kwargs)
            if "x" in mode:
                value = self.label_path(path)
                self.emit("create:" + value if value else None)
            return stream

        def replaced(first: Any, second: Any, *args: Any, **kwargs: Any) -> None:
            original_replace(first, second, *args, **kwargs)
            value = self.label_path(second)
            if value:
                data = fs.read_json(Path(second))
                phase = str(data.get("phase", "committed"))
                if Path(second).parent.name == "checkpoint-file-operations":
                    phase += f"/published={len(data.get('published', []))}/archived={len(data.get('archived', []))}"
                self.emit("json:" + Path(second).parent.name + ":" + phase)

        def unlinked(path: Any, *args: Any, **kwargs: Any) -> None:
            value = self.label_path(path, kwargs.get("dir_fd"))
            original_unlink(path, *args, **kwargs)
            self.emit("unlink:" + value if value else None)

        def removed(path: Any, *args: Any, **kwargs: Any) -> None:
            value = self.label_path(path, kwargs.get("dir_fd"))
            original_rmdir(path, *args, **kwargs)
            self.emit("rmdir:" + value if value else None)

        def synced(fd: int) -> None:
            original_fsync(fd)
            if fd == self.fd or not stat.S_ISREG(os.fstat(fd).st_mode):
                return
            path = Path(os.readlink(f"/proc/self/fd/{fd}"))
            value = self.label_path(path)
            if value and path.suffix != ".jsonl":
                self.emit("fsync-file:" + value)

        def linked(first: Any, second: Any, **kwargs: Any) -> None:
            original_link(first, second, **kwargs)
            a = self.label_path(first, kwargs.get("src_dir_fd"))
            b = self.label_path(second, kwargs.get("dst_dir_fd"))
            self.emit(f"link:{a}->{b}" if a or b else None)

        def chmod(path: Any, *args: Any, **kwargs: Any) -> None:
            original_chmod(path, *args, **kwargs)
            value = self.label_path(path, kwargs.get("dir_fd"))
            self.emit("chmod:" + value if value else None)

        def utime(path: Any, *args: Any, **kwargs: Any) -> None:
            original_utime(path, *args, **kwargs)
            value = self.label_path(path, kwargs.get("dir_fd"))
            self.emit("utime:" + value if value else None)

        def renamed(first: Path, second: Path, **kwargs: Any) -> None:
            original_rename(first, second, **kwargs)
            a, b = self.label_path(first), self.label_path(second)
            self.emit(f"rename:{a}->{b}")

        def claimed(first_fd: int, first: str, second_fd: int, second: str) -> None:
            a, b = self.label_path(first, first_fd), self.label_path(second, second_fd)
            original_claim(first_fd, first, second_fd, second)
            self.emit(f"claim:{a}->{b}")

        def appended(path: Path, value: dict[str, Any]) -> None:
            original_append(path, value)
            role = "audit" if path.name == "audit.jsonl" else path.parent.name
            self.emit(
                f"append:{role}:{value['event']}"
                + (":" + value["entry"] if "entry" in value else "")
            )

        for owner, name, primitive in (
            (Path, "mkdir", mkdir),
            (Path, "open", opened),
            (os, "link", linked),
            (os, "chmod", chmod),
            (os, "utime", utime),
            (os, "replace", replaced),
            (os, "unlink", unlinked),
            (os, "rmdir", removed),
            (os, "fsync", synced),
        ):
            setattr(owner, name, primitive)
        for module_name in ("filesystem", "engine", "publication", "checkpoint_files"):
            module = importlib.import_module("storage_tier." + module_name)
            for name, original, replacement in (
                ("rename", original_rename, renamed),
                ("rename_at", original_claim, claimed),
                ("append_json", original_append, appended),
            ):
                if getattr(module, name, None) is original:
                    setattr(module, name, replacement)


def run_child(
    lab: Lab,
    action: Callable[[], Any],
    output: Path,
    *,
    stop: int | None = None,
    stop_when: Callable[[str], bool] | None = None,
) -> tuple[int, list[str]]:
    def child() -> None:
        try:
            Trace(lab, output, stop, stop_when).install()
            outcome = action()
            output.with_suffix(".result.json").write_text(json.dumps(outcome))
        except BaseException:
            traceback.print_exc()
            os._exit(92)
        os._exit(0)

    process = multiprocessing.get_context("fork").Process(target=child)
    process.start()
    process.join(timeout=15)
    if process.is_alive():
        process.kill()
        process.join()
        raise AssertionError("storage boundary child exceeded 15 seconds")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    code = process.exitcode
    assert code is not None
    return code, rows


def payloads(lab: Lab) -> set[bytes]:
    found = set()
    for base in (lab.config.repo, lab.config.mirror):
        for path in base.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = str(path.relative_to(base))
            if relative.startswith(".storage-tier/") and not any(
                v in relative
                for v in (
                    "checkpoint-file-history/",
                    "recovered-assets/",
                    "recovered-trees/",
                    "recovered-publications/",
                    "manual-repair/",
                    "manual-recovery/",
                )
            ):
                continue
            found.add(lab.filesystem.read_bytes(path))
    return found


def finish_recovery(lab: Lab, unit: Any) -> None:
    before = payloads(lab)
    code, recovered = lab.engine.run("recover", [], dry_run=False)
    if lab.orphan is not None and lab.orphan.exists():
        assert code == 1 and "unclaimed storage artifacts" in recovered["error"], (
            recovered
        )
        archive = lab.config.mirror / ".storage-tier/manual-recovery" / lab.orphan.name
        lab.engine.ensure_parent(archive.parent, lab.config.mirror)
        identity = lab.orphan.stat().st_dev, lab.orphan.stat().st_ino
        lab.filesystem.rename(lab.orphan, archive)
        assert identity == (archive.stat().st_dev, archive.stat().st_ino)
        code, recovered = lab.engine.run("recover", [], dry_run=False)
    assert code == 0, recovered
    assert before <= payloads(lab), "recovery discarded retained payload"
    assert not lab.layout.discover(lab.config, lab.config.repo).errors
    assert not lab.layout.discover(lab.config, lab.config.mirror).errors
    source, destination = lab.config.repo / unit.path, lab.config.mirror / unit.path
    if unit.kind == "checkpoint-files":
        saved = lab.engine.manifest_path(unit)
        if (
            saved.exists()
            and lab.filesystem.read_json(saved).get("kind") == "checkpoint"
        ):
            assert lab.run("index", unit)[0] == 0
    if lab.engine.recovery_path(unit).exists():
        if (
            source.exists()
            and lab.filesystem.snapshot(source, hashes=True).content()
            == lab.filesystem.snapshot(destination, hashes=True).content()
        ):
            assert lab.run("verify", unit)[0] == 0
        else:
            if source.exists():
                archive = lab.config.state / "manual-repair" / "partial-unit"
                archive.parent.mkdir(parents=True, exist_ok=True)
                lab.filesystem.rename(source, archive)
            assert lab.run("hydrate", unit)[0] == 0
    elif not source.exists() or (
        unit.kind == "checkpoint-files"
        and not any(
            e["kind"] == "file"
            for e in lab.filesystem.snapshot(
                source, root_files_only=True
            ).entries.values()
        )
    ):
        assert lab.run("hydrate", unit)[0] == 0
    else:
        assert lab.run("mirror", unit)[0] == 0
    assert lab.engine.run("recover", [], dry_run=False)[1]["results"] == []
    code, report = lab.engine.run("cycle", [], dry_run=False)
    assert code == 0, report
    assert not report["layout_errors"]
    if lab.wanted is not None:
        actual = {
            str(p.relative_to(destination)): lab.filesystem.read_bytes(p)
            for p in destination.rglob("*")
            if p.is_file()
        }
        assert actual == lab.wanted, "daily cycle lost the intended canonical payload"
