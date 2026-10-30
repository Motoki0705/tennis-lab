"""Advance an explicit owned-session plan only after verified terminal jobs."""

import json
import subprocess
import sys
import time
from pathlib import Path

plan_path = Path(sys.argv[1])
plan = json.loads(plan_path.read_text())
directory = Path(sys.argv[2])
directory.mkdir(parents=True, exist_ok=True)
with (directory / 'controller-started.json').open('x') as stream:
    json.dump(dict(pid=__import__('os').getpid(), plan=str(plan_path)), stream)
cli = ['.venv/bin/python', '.agents/skills/tennis-colab/scripts/colab.py']

for item in plan['jobs']:
    job = item['job']
    if item['argv'] is not None:
        submitted = subprocess.run(item['argv'], capture_output=True, text=True, check=True)
        (directory / (job + '-submission.json')).write_text(submitted.stdout)
        print(json.dumps(dict(job=job, status='submitted')), flush=True)
    while True:
        response = subprocess.run(cli + ['status', '--session', plan['session'], '--job-id', job],
                                  capture_output=True, text=True, check=True)
        value = json.loads(response.stdout)
        (directory / (job + '-status.json')).write_text(response.stdout)
        state = value['jobs'][0]
        if state['status'] in {'completed', 'failed', 'cancelled', 'save_failed', 'unknown', 'dispatch_failed', 'abandoned'}:
            print(json.dumps(dict(job=job, **state)), flush=True)
            if (state['status'] != ('completed' if item['expected'] == 0 else 'failed')
                    or state.get('returncode') != item['expected'] or not state.get('drive_saved_at')):
                raise RuntimeError('Plan stopped: unexpected result or missing durable-save receipt')
            break
        time.sleep(30)
(directory / 'controller-completed.json').write_text(json.dumps(dict(status='completed', jobs=[item['job'] for item in plan['jobs']])) + '\n')
