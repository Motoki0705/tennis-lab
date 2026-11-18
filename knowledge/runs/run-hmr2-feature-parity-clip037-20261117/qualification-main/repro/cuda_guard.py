"""Bound NVML health probes and preserve diagnostics on renewed driver failure."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Iterator

PROBE = ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']


def probe() -> str:
    result = subprocess.run(PROBE, capture_output=True, text=True, timeout=10, check=True)
    return result.stdout.strip()


@contextmanager
def guard(output: Path, operation: str) -> Iterator[None]:
    output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    def check() -> None:
        now = datetime.now(UTC).isoformat()
        try:
            status = probe()
        except BaseException as error:
            record = {'timestamp_utc': now, 'operation': operation, 'command': PROBE,
                'type': type(error).__name__, 'error': str(error),
                'stdout': str(getattr(error, 'stdout', '')), 'stderr': str(getattr(error, 'stderr', ''))}
            (output / 'cuda_health_failure.json').write_text(json.dumps(record, indent=2))
            with (output / 'cuda_health_failure_threads.txt').open('w') as stack:
                faulthandler.dump_traceback(file=stack, all_threads=True)
            print(json.dumps(record), flush=True)
            os._exit(86)  # Exit this owned queue workload; do not retry a broken driver.
        with (output / 'cuda_health.jsonl').open('a') as log:
            log.write(json.dumps({'timestamp_utc': now, 'operation': operation, 'status': status}) + '\n')
    check()  # Probe before creating any CUDA context.
    def watch() -> None:
        while not stop.wait(60):
            check()
    thread = threading.Thread(target=watch, name='cuda-health-guard', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=12)
        check()
