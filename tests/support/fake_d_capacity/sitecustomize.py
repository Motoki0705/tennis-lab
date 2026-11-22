"""Supply a synthetic DrvFs mount and capacity to subprocess tests."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

BYTES_FILE = "TENNIS_TEST_D_CAPACITY_BYTES_FILE"
D_MOUNTINFO = r"900000 1 0:72 / /mnt/d rw - 9p D:\134 rw,aname=drvfs;path=D:\;"
_read_text = Path.read_text
_statvfs = os.statvfs


def install(setattr_: Callable[[Any, str, Any], None] = setattr) -> None:
    available_file = Path(os.environ[BYTES_FILE])

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        text = _read_text(path, *args, **kwargs)
        if path != Path("/proc/self/mountinfo"):
            return text
        return "\n".join(
            [
                *(line for line in text.splitlines() if line.split()[4] != "/mnt/d"),
                D_MOUNTINFO,
            ]
        )

    def statvfs(path: Any) -> os.statvfs_result:
        if path != "/mnt/d":
            return _statvfs(path)
        available = int(_read_text(available_file))
        return os.statvfs_result((4096, 1, 0, 0, available, 0, 0, 0, 0, 255))

    setattr_(Path, "read_text", read_text)
    setattr_(os, "statvfs", statvfs)


if __name__ == "sitecustomize" and BYTES_FILE in os.environ:
    install()
