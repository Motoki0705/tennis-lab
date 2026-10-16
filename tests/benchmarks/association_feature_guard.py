"""Supervise only this feature job; fail closed on resource-monitor failures."""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]


def gpu_bytes() -> int:
    value = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '--id=0'],
        text=True, timeout=10).strip()
    if not value.isdigit():
        raise RuntimeError(f'Cannot monitor GPU VRAM: {value!r}')
    return int(value) * 1024**2


def supervise(report: Path, command: list[str]) -> int:
    plan = json.loads((report / 'plan.json').read_text())
    budget = plan['budget']
    receipt: dict[str, Any] = {'status': 'running', 'command': command, 'sample_interval_s': 2.,
                             'measurement': 'NVML via nvidia-smi, entire GPU 0; conservative for an all job',
                             'peak_gpu_bytes': 0, 'min_host_available_bytes': psutil.virtual_memory().available}
    path = report / 'resource_guard.json'
    if path.exists():
        raise FileExistsError(path)
    started = time.monotonic()
    child: subprocess.Popen[bytes] | None = None

    def save() -> None:
        receipt['wall_seconds'] = time.monotonic() - started
        pending = path.with_suffix('.tmp')
        pending.write_text(json.dumps(receipt, indent=2) + '\n')
        pending.replace(path)

    def stop(signum: int, frame: Any) -> None:
        raise RuntimeError(f'Feature supervisor received signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            used = gpu_bytes()
            available = psutil.virtual_memory().available
            size = 0
            for item in report.rglob('*'):
                try:
                    if item.is_file():
                        size += item.stat().st_size
                except FileNotFoundError:
                    # The producer atomically replaces its small progress JSON.
                    continue
            receipt['peak_gpu_bytes'] = max(receipt['peak_gpu_bytes'], used)
            receipt['min_host_available_bytes'] = min(receipt['min_host_available_bytes'], available)
            receipt['output_bytes'] = size
            save()
            if used >= budget['vram_stop_bytes']:
                raise RuntimeError('GPU VRAM reached the precommitted stop limit')
            if available < 6 * 1024**3 or size >= budget['disk_limit_bytes']:
                raise RuntimeError('Host RAM or output disk budget exhausted')
            if time.monotonic() - started >= budget['wall_seconds'] - 15:
                raise RuntimeError('Feature job reached its wall-time limit')
            if child is None:
                child = subprocess.Popen(command)
            code = child.poll()
            if code is not None:
                receipt.update(status='ok' if code == 0 else 'failed', exit_code=code)
                save()
                return code
            time.sleep(2)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        receipt.update(status='failed', error=str(error))
        save()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a child command is required')
    raise SystemExit(supervise(args.report.resolve(), command))


if __name__ == '__main__':
    main()
