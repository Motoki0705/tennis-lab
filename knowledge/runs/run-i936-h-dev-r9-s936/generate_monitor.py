"""Run the pre-registered CPU generation once, with campaign resource limits."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

project, data, output, logs = map(Path, sys.argv[1:])
logs.mkdir(parents=True, exist_ok=False)
command = [str(project / '.venv/bin/python'), '-u', '-m',
           'src.tasks.ball_refiner.scripts.generate_synthetic_3d',
           '--project-root', str(project), '--data-root', str(data),
           '--plan', str(project / 'src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml'),
           '--output', str(output), '--mode', 'dev', '--calibration-report',
           str(project / 'knowledge/runs/run-i936-provisional-degradation-r5-s936/calibration.json')]
environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
started = time.monotonic()
receipt = {'command': command, 'cwd': str(project), 'output': str(output),
           'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip(),
           'started_unix': time.time(), 'maximum_seconds': 5400, 'maximum_bytes': 750000000,
           'minimum_available_bytes': 6 * 1024**3, 'status': 'running'}
(logs / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
with (logs / 'generation.log').open('w') as log, (logs / 'resources.jsonl').open('w') as samples:
    process = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(logs / 'time.txt'), *command],
                               cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    receipt['pid'] = process.pid
    reason = None
    while process.poll() is None:
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        size = sum(p.stat().st_size for p in output.glob('*') if p.is_file())
        elapsed = time.monotonic() - started
        sample = {'elapsed_seconds': elapsed, 'available_bytes': available, 'output_bytes': size}
        samples.write(json.dumps(sample) + '\n')
        samples.flush()
        if elapsed > 5400:
            reason = '90-minute deadline'
        elif available < 6 * 1024**3:
            reason = 'available RAM below 6 GiB'
        elif size > 750000000:
            reason = 'disk reservation exceeded'
        if reason:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            break
        time.sleep(10)
    receipt.update(status='complete' if process.returncode == 0 else 'failed',
                   exit_code=process.returncode, stop_reason=reason,
                   elapsed_seconds=time.monotonic() - started)
(logs / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
