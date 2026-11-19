"""Out-of-process CUDA/NVML liveness guard. A failure never triggers a retry."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from contextlib import suppress
from datetime import UTC, datetime

NVIDIA_SMI = '/usr/lib/wsl/lib/nvidia-smi'
QUERY_TIMEOUT_SECONDS = 10.0
HEARTBEAT_TIMEOUT_SECONDS = 180.0
SAMPLE_INTERVAL_SECONDS = 1.0


def utc():
    return datetime.now(UTC).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def write_anomaly(root, value):
    """Atomically keep the first anomaly from either the child or guardian."""
    target = root / 'anomaly.json'
    temporary = root / f'.anomaly.{os.getpid()}.tmp'
    temporary.write_text(json.dumps({'at_utc': utc(), **value}, ensure_ascii=False, indent=2) + '\n')
    try:
        os.link(temporary, target)
    except FileExistsError:
        pass
    finally:
        temporary.unlink()


class HealthError(RuntimeError):
    def __init__(self, kind, detail):
        self.kind, self.detail = kind, detail
        super().__init__(detail)


def query_health(binary=NVIDIA_SMI, *, timeout=QUERY_TIMEOUT_SECONDS):
    command = [binary, '--id=0', '--query-gpu=memory.used,driver_version', '--format=csv,noheader,nounits']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Bound the reap too: an uninterruptible driver query must not hide the timeout.
        detail = {'command': command, 'timeout_seconds': timeout, 'pid': process.pid,
                  'stdout': str(exc.output or ''), 'stderr': str(exc.stderr or '')}
        process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            detail['termination_pending'] = True
        raise HealthError('nvidia_smi_unresponsive', json.dumps(detail, ensure_ascii=False)) from exc
    if process.returncode != 0:
        raise HealthError('nvidia_smi_error', json.dumps({'command': command,
            'returncode': process.returncode, 'stdout': stdout, 'stderr': stderr}, ensure_ascii=False))
    try:
        memory, driver = stdout.strip().split(',')
        return int(memory.strip()), driver.strip()
    except ValueError as exc:
        raise HealthError('nvidia_smi_invalid_response', json.dumps({'command': command,
            'stdout': stdout, 'stderr': stderr}, ensure_ascii=False)) from exc


def capture_diagnostics(root, name, process, *, binary=NVIDIA_SMI,
                        query_timeout=QUERY_TIMEOUT_SECONDS, stack_timeout=5.0):
    """Save all-thread stacks and one bounded GPU report before sending TERM."""
    health = root / 'health'
    stack = health / f'{name}-threads.log'
    record = {'started_at_utc': utc(), 'pid': process.pid, 'stack_file': str(stack)}
    if process.poll() is None:
        try:
            os.kill(process.pid, signal.SIGUSR1)
            record['stack_signal_at_utc'] = utc()
        except ProcessLookupError:
            record['stack_signal_error'] = 'process exited'
    deadline = time.monotonic() + stack_timeout
    while time.monotonic() < deadline and process.poll() is None:
        if stack.exists() and 'Thread' in stack.read_text(errors='replace'):
            break
        time.sleep(0.02)
    record['stack_obtained_before_sigterm'] = stack.exists() and stack.stat().st_size > 0
    command = [binary, '--id=0', '-q', '-d', 'PERFORMANCE,CLOCK,POWER,TEMPERATURE']
    record.update(command=command, query_started_at_utc=utc())
    query = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = query.communicate(timeout=query_timeout)
        record['query_returncode'] = query.returncode
    except subprocess.TimeoutExpired as exc:
        record.update(query_timeout_seconds=query_timeout, query_pid=query.pid)
        query.kill()
        try:
            stdout, stderr = query.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            stdout, stderr = exc.output or '', exc.stderr or ''
            record['query_termination_pending'] = True
    stdout = stdout.decode(errors='replace') if isinstance(stdout, bytes) else stdout
    stderr = stderr.decode(errors='replace') if isinstance(stderr, bytes) else stderr
    (health / f'{name}-nvidia-smi-q.stdout.log').write_text(stdout)
    (health / f'{name}-nvidia-smi-q.stderr.log').write_text(stderr)
    record.update(finished_at_utc=utc(), process_alive_before_sigterm=process.poll() is None,
                  stack_bytes_before_sigterm=stack.stat().st_size if stack.exists() else 0)
    write(health / f'{name}-diagnostics.json', record)
    return record


def terminate_variant(process):
    if process.poll() is not None:
        return False
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            return True  # Queue wrapper retains ownership until verified teardown.
    return False


def run_variant(root, name, command, cwd, *, query=query_health,
                heartbeat_timeout=HEARTBEAT_TIMEOUT_SECONDS, interval=SAMPLE_INTERVAL_SECONDS,
                diagnose=capture_diagnostics):
    """Only liveness/terminal monitoring; no model parameter or CUDA mode changes."""
    health_dir = root / 'health'
    health_dir.mkdir(exist_ok=True)
    result = {'variant': name, 'started_at_utc': utc(), 'status': 'running',
              'device_memory_sampling_ms': int(interval * 1000), 'device_memory_samples': 0,
              'query_timeout_seconds': QUERY_TIMEOUT_SECONDS,
              'heartbeat_timeout_seconds': heartbeat_timeout}
    process = None
    samples = []
    started = time.monotonic()
    log_path = root / f'{name}.log'
    def stop_after_diagnostics():
        try:
            result['diagnostics'] = diagnose(root, name, process)
        except Exception as diagnostic_error:
            result['diagnostic_error'] = repr(diagnostic_error)
        finally:
            result['termination_pending'] = terminate_variant(process)
            result['returncode'] = process.poll()
    try:
        # Do not start model execution if the first health check fails.
        memory, driver = query()
        samples.append(memory)
        result['driver'] = driver
        with log_path.open('x') as log, (health_dir / f'{name}-memory.csv').open('x') as csv:
            csv.write(f'{utc()},{memory}\n')
            csv.flush()
            env = {**os.environ, 'TENNIS_STACK_FILE': str(health_dir / f'{name}-threads.log')}
            process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env)
            result['pid'] = process.pid
            while process.poll() is None:
                if (root / 'anomaly.json').exists():
                    stop_after_diagnostics()
                    break
                heartbeat = root / name / 'heartbeat.json'
                stamp = json.loads(heartbeat.read_text()) if heartbeat.exists() else None
                last = started if stamp is None else stamp['monotonic']
                if time.monotonic() - last > heartbeat_timeout:
                    raise HealthError('worker_heartbeat_timeout', json.dumps({
                        'pid': process.pid, 'timeout_seconds': heartbeat_timeout,
                        'last_heartbeat': stamp, 'command': command}, ensure_ascii=False))
                memory, driver = query()
                samples.append(memory)
                csv.write(f'{utc()},{memory}\n')
                csv.flush()
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=interval)
        result['returncode'] = process.poll()
        result['status'] = 'complete' if process.returncode == 0 and not (root / 'anomaly.json').exists() else 'failed'
        if result['status'] == 'failed' and not (root / 'anomaly.json').exists():
            detail = log_path.read_text()
            marker = detail.rfind('Traceback (most recent call last):')
            error_detail = detail[marker:] if marker >= 0 else detail[-8192:]
            if (process.returncode is not None and process.returncode < 0) or any(
                word in error_detail.lower() for word in ('cuda', 'cublas', 'cudnn', 'nvml', 'illegal memory access')
            ):
                write_anomaly(root, {'kind': 'cuda_process_failure', 'variant': name,
                    'detail': detail, 'log': str(log_path), 'pid': process.pid})
    except HealthError as exc:
        write_anomaly(root, {'kind': exc.kind, 'variant': name, 'detail': exc.detail,
                            'log': str(log_path), 'pid': None if process is None else process.pid})
        result.update(status='failed', error_kind=exc.kind, error=exc.detail)
        if process is not None:
            stop_after_diagnostics()
    finally:
        result.update(finished_at_utc=utc(), device_memory_samples=len(samples),
                      device_memory_peak_mib=max(samples) if samples else None,
                      device_memory_min_mib=min(samples) if samples else None)
        write(health_dir / f'{name}.json', result)
    return result
