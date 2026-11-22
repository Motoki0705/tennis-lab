"""Wait for the queue's verified terminal publication using filesystem events."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import select
import sys

from src.utils.training_queue import read_completion

queue, job = Path(sys.argv[1]), sys.argv[2]
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
if fd < 0:
    raise OSError(ctypes.get_errno(), 'inotify_init1')
try:
    wd = libc.inotify_add_watch(fd, os.fsencode(queue / 'control/completions'), 8 | 128 | 256)
    if wd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_add_watch')
    while True:
        record = read_completion(queue, job)
        if record is not None:
            print(json.dumps(record), flush=True)
            break
        select.select([fd], [], [])
        try:
            os.read(fd, 65536)
        except BlockingIOError:
            pass
finally:
    os.close(fd)
