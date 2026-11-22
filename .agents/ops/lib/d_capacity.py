"""Measure physical Windows D: and check whether a new workload may start."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

D_MOUNT = "/mnt/d"
MOUNTINFO = Path("/proc/self/mountinfo")
BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")
CONFIG_PATH = Path("/etc/tennis-lab-d-capacity.toml")
CONFIG_ENV = "TENNIS_D_CAPACITY_CONFIG"
STATE_PATH = Path("/run/tennis-lab-d-capacity/status.json")
GIB = 1 << 30
DEFAULT_THRESHOLD_GIB = 20
MAX_STATUS_AGE_NS = 90 * 10**9
REFUSED = 75
Decision = Literal["allow", "blocked"]


class CapacityError(ValueError):
    """A new workload cannot be admitted on the available evidence."""


def _physical_mount_id() -> str:
    matches = []
    for line in MOUNTINFO.read_text(
        encoding="utf-8", errors="surrogateescape"
    ).splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[4] == D_MOUNT:
            matches.append(fields)
    if len(matches) == 1:
        fields = matches[0]
        filesystem, source, options = fields[fields.index("-", 6) + 1 :]
        if (
            fields[3] == "/"
            and source.lower() in (r"d:\134", "d:")
            and (
                filesystem == "drvfs"
                or (
                    filesystem == "9p"
                    and any(
                        option.startswith("aname=drvfs;")
                        for option in options.split(",")
                    )
                )
            )
        ):
            return fields[0]
    raise CapacityError(f"{D_MOUNT} is not mounted as the Windows D: drive root")


def measure_available_bytes() -> int:
    """Read the Windows volume's available bytes from its DrvFs mount."""
    try:
        mount_id = _physical_mount_id()
        info = os.statvfs(D_MOUNT)
        if _physical_mount_id() != mount_id:
            raise CapacityError(
                f"the Windows D: mount at {D_MOUNT} changed during measurement"
            )
    except CapacityError:
        raise
    except (OSError, ValueError) as exc:
        raise CapacityError(f"cannot measure Windows D: at {D_MOUNT}: {exc}") from exc
    if info.f_bavail < 0 or info.f_frsize <= 0:
        raise CapacityError(f"invalid statvfs values for Windows D: at {D_MOUNT}")
    return info.f_bavail * info.f_frsize


def threshold_bytes() -> int:
    configured = os.environ.get(CONFIG_ENV)
    path = CONFIG_PATH if configured is None else Path(configured)
    try:
        with path.open("rb") as stream:
            values = tomllib.load(stream)
    except FileNotFoundError as exc:
        if configured is None:
            return DEFAULT_THRESHOLD_GIB * GIB
        raise CapacityError(f"D: capacity config {path} is missing") from exc
    except (OSError, ValueError) as exc:
        raise CapacityError(f"cannot read D: capacity config {path}: {exc}") from exc
    value = values.get("threshold_gib")
    if values.keys() != {"threshold_gib"} or type(value) is not int or value <= 0:
        raise CapacityError(
            f"D: capacity config {path} requires a positive integer threshold_gib"
        )
    return value * GIB


def decide(available: int, threshold: int) -> Decision:
    return "blocked" if available < threshold else "allow"


@dataclass(frozen=True)
class CapacityStatus:
    measured_at_ns: int
    boot_id: str
    boottime_ns: int
    available_bytes: int | None
    threshold_bytes: int | None
    decision: Literal["allow", "blocked", "error"]
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps({"schema_version": 2, **asdict(self)}, sort_keys=True) + "\n"

    def require_allowed(self) -> None:
        if self.decision == "error":
            raise CapacityError(f"D: capacity measurement failed: {self.error}")
        if self.decision == "blocked":
            raise CapacityError(
                f"physical D: has {self.available_bytes} bytes available, below the "
                f"{self.threshold_bytes}-byte threshold"
            )


def boot_clock() -> tuple[str, int]:
    try:
        boot_id = str(uuid.UUID(BOOT_ID_FILE.read_text(encoding="ascii").strip()))
        return boot_id, time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    except (OSError, ValueError) as exc:
        raise CapacityError(f"cannot read the D: watchdog boot clock: {exc}") from exc


def sample() -> CapacityStatus:
    boot_id, boottime = boot_clock()
    measured_at = time.time_ns()
    threshold = None
    try:
        threshold = threshold_bytes()
        available = measure_available_bytes()
    except CapacityError as exc:
        return CapacityStatus(
            measured_at, boot_id, boottime, None, threshold, "error", str(exc)
        )
    return CapacityStatus(
        measured_at,
        boot_id,
        boottime,
        available,
        threshold,
        decide(available, threshold),
    )


def read_status(path: Path) -> CapacityStatus:
    """Read and validate the schema of a published status."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise CapacityError(f"cannot read D: watchdog status {path}: {exc}") from exc
    try:
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
            raise ValueError
        if data.pop("schema_version") != 2:
            raise ValueError
        status = CapacityStatus(**data)
        if (
            type(status.measured_at_ns) is not int
            or not isinstance(status.boot_id, str)
            or str(uuid.UUID(status.boot_id)) != status.boot_id
            or type(status.boottime_ns) is not int
            or status.boottime_ns < 0
            or (
                status.threshold_bytes is not None
                and (
                    type(status.threshold_bytes) is not int
                    or status.threshold_bytes <= 0
                )
            )
        ):
            raise ValueError
        if status.decision == "error":
            if (
                status.available_bytes is not None
                or not isinstance(status.error, str)
                or not status.error
            ):
                raise ValueError
        elif (
            type(status.available_bytes) is not int
            or status.available_bytes < 0
            or status.threshold_bytes is None
            or status.decision != decide(status.available_bytes, status.threshold_bytes)
            or status.error is not None
        ):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise CapacityError(f"invalid D: watchdog status in {path}") from exc
    return status


def require_capacity(state_file: Path | None = None) -> None:
    if state_file is None:
        sample().require_allowed()
        return
    try:
        status = read_status(state_file)
    except FileNotFoundError as exc:
        raise CapacityError(f"D: watchdog status is missing: {state_file}") from exc
    boot_id, boottime = boot_clock()
    if status.boot_id != boot_id:
        raise CapacityError(
            f"D: watchdog status is from a different boot: {state_file}"
        )
    age_ns = boottime - status.boottime_ns
    if age_ns < 0:
        raise CapacityError(
            f"D: watchdog status has a future measurement time: {state_file}"
        )
    if age_ns > MAX_STATUS_AGE_NS:
        raise CapacityError(
            f"D: watchdog status is older than 90 seconds: {state_file}"
        )
    status.require_allowed()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, help="use only this watchdog status")
    args = parser.parse_args()
    try:
        require_capacity(args.state_file)
    except CapacityError as exc:
        print(f"D: launch refused: {exc}", file=sys.stderr)
        return REFUSED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
