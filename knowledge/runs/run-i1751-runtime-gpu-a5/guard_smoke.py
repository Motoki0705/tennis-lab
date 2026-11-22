"""CPU-only checks of first-error stopping, bounded health queries and profiling."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import subprocess
import time
from pathlib import Path

from health_guard import HealthError, capture_diagnostics, query_health, run_variant as guarded_variant, terminate_variant
from run_pair import execute_pair


def run_variant(*args, **kwargs):
    kwargs.setdefault('diagnose', lambda *a: {'cpu_fake': True})
    return guarded_variant(*args, **kwargs)


STACK_CHILD = '''import faulthandler, os, signal, threading
f = open(os.environ['TENNIS_STACK_FILE'], 'a')
faulthandler.enable(file=f, all_threads=True)
faulthandler.register(signal.SIGUSR1, file=f, all_threads=True)
faulthandler.dump_traceback_later(.05, file=f)
def worker_wait():
    threading.Event().wait(10)
threading.Thread(target=worker_wait, daemon=True).start()
print('ready', flush=True)
def main_wait():
    threading.Event().wait(10)
main_wait()
'''


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cases = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        success = root / 'fake-nvidia-success'
        success.write_text('#!/bin/sh\nprintf "12, 621.24\\n"\n')
        success.chmod(0o700)
        assert query_health(str(success)) == (12, '621.24')
        cases.append('health_query_success')
        slow = root / 'fake-nvidia-timeout'
        slow.write_text('#!/bin/sh\nexec sleep 1\n')
        slow.chmod(0o700)
        try:
            query_health(str(slow), timeout=0.02)
        except HealthError as exc:
            assert exc.kind == 'nvidia_smi_unresponsive'
            assert 'timeout_seconds' in exc.detail
        else:
            raise AssertionError('An unresponsive query was accepted')
        cases.append('query_timeout_stops_without_retry')
        failure = root / 'fake-nvidia-error'
        failure.write_text('#!/bin/sh\nprintf "Failed to initialize NVML: fake error\\n" >&2\nexit 3\n')
        failure.chmod(0o700)
        try:
            query_health(str(failure))
        except HealthError as exc:
            assert exc.kind == 'nvidia_smi_error' and 'fake error' in exc.detail
        else:
            raise AssertionError('NVML error was ignored')
        cases.append('nvml_error_keeps_full_stderr')
        normal_root = root / 'normal'
        normal_root.mkdir()
        outcome = run_variant(normal_root, 'main', [sys.executable, '-c', 'import time; time.sleep(.05)'],
                              Path.cwd(), query=lambda: (12, 'fake'), interval=0.02)
        assert outcome['status'] == 'complete' and outcome['returncode'] == 0
        cases.append('normal_child_completes')
        application_root = root / 'application-error'
        application_root.mkdir()
        outcome = run_variant(application_root, 'main', [sys.executable, '-c',
            'print(\'device=cuda\'); print(\'Traceback (most recent call last):\'); print(\'ValueError: ordinary input failure\'); raise SystemExit(1)'],
            Path.cwd(), query=lambda: (12, 'fake'), interval=0.02)
        assert outcome['status'] == 'failed' and not (application_root / 'anomaly.json').exists()
        cases.append('ordinary_failure_is_not_mislabelled_by_cuda_metadata')
        cuda_root = root / 'cuda-error'
        cuda_root.mkdir()
        outcome = run_variant(cuda_root, 'main', [sys.executable, '-c',
            'print(\'Traceback (most recent call last):\'); print(\'RuntimeError: CUDA error: fake unknown error\'); raise SystemExit(1)'],
            Path.cwd(), query=lambda: (12, 'fake'), interval=0.02)
        assert outcome['status'] == 'failed'
        assert 'fake unknown error' in json.loads((cuda_root / 'anomaly.json').read_text())['detail']
        cases.append('first_cuda_failure_is_recorded')
        hang_root = root / 'hang'
        hang_root.mkdir()
        outcome = run_variant(hang_root, 'main', [sys.executable, '-c', 'import time; time.sleep(2)'],
                              Path.cwd(), query=lambda: (12, 'fake'), heartbeat_timeout=0.06, interval=0.02)
        assert outcome['status'] == 'failed'
        assert json.loads((hang_root / 'anomaly.json').read_text())['kind'] == 'worker_heartbeat_timeout'
        cases.append('stalled_child_records_time_and_stops')
        failed_diagnostic_root = root / 'diagnostic-exception'
        failed_diagnostic_root.mkdir()
        def bad_diagnostic(*args):
            raise OSError('fake diagnostic tool unavailable')
        outcome = run_variant(failed_diagnostic_root, 'main',
            [sys.executable, '-c', 'import time; time.sleep(2)'], Path.cwd(),
            query=lambda: (12, 'fake'), heartbeat_timeout=.06, interval=.02, diagnose=bad_diagnostic)
        assert 'fake diagnostic tool unavailable' in outcome['diagnostic_error']
        assert outcome['returncode'] == -15 and outcome['termination_pending'] is False
        cases.append('diagnostic_exception_still_terminates_child')
        seen = []
        def fail_run(root, name, command, cwd):
            seen.append(name)
            return {'status': 'failed', 'returncode': 1}
        outputs = execute_pair(root, {'main': (root, 'a'), 'runtime': (root, 'b')}, run=fail_run)
        assert seen == ['main'] and set(outputs) == {'main'}
        cases.append('no_runtime_process_after_main_failure')
        diagnostic_root = root / 'diagnostic-hang'
        diagnostic_root.mkdir()
        diagnostic_binary = root / 'fake-diagnostic'
        diagnostic_binary.write_text('#!/bin/sh\nprintf "PERFORMANCE CLOCK POWER TEMPERATURE fake\\n"\nprintf "diagnostic stderr\\n" >&2\n')
        diagnostic_binary.chmod(0o700)
        def diagnose(directory, name, process):
            return capture_diagnostics(directory, name, process, binary=str(diagnostic_binary),
                                       query_timeout=.2, stack_timeout=.5)
        outcome = run_variant(diagnostic_root, 'main', [sys.executable, '-u', '-c', STACK_CHILD],
                              Path.cwd(), query=lambda: (12, 'fake'), heartbeat_timeout=.2,
                              interval=.02, diagnose=diagnose)
        d = outcome['diagnostics']
        stack = Path(d['stack_file']).read_text()
        assert 'worker_wait' in stack and 'main_wait' in stack
        assert d['stack_obtained_before_sigterm'] and d['process_alive_before_sigterm']
        assert d['command'][-3:] == ['-q', '-d', 'PERFORMANCE,CLOCK,POWER,TEMPERATURE']
        assert outcome['returncode'] == -15
        cases.append('all_thread_stacks_and_gpu_report_precede_sigterm')
        assert 'PERFORMANCE CLOCK POWER TEMPERATURE fake' in (diagnostic_root/'health/main-nvidia-smi-q.stdout.log').read_text()
        assert 'diagnostic stderr' in (diagnostic_root/'health/main-nvidia-smi-q.stderr.log').read_text()
        cases.append('diagnostic_full_stdout_and_stderr_are_saved')
        for label, contents in [('error', '#!/bin/sh\nprintf "full NVML diagnostic error\\n" >&2\nexit 19\n'),
                                ('timeout', '#!/bin/sh\nexec sleep 1\n')]:
            directory = root / ('diagnostic-'+label)
            (directory/'health').mkdir(parents=True)
            binary = root / ('fake-diagnostic-'+label)
            binary.write_text(contents)
            binary.chmod(0o700)
            env = {**os.environ, 'TENNIS_STACK_FILE': str(directory/'health/main-threads.log')}
            child = subprocess.Popen([sys.executable, '-u', '-c', STACK_CHILD], env=env,
                                     stdout=subprocess.PIPE, text=True)
            assert child.stdout.readline().strip() == 'ready'
            started = time.monotonic()
            try:
                report = capture_diagnostics(directory, 'main', child, binary=str(binary),
                                             query_timeout=.02, stack_timeout=.3)
                assert report['process_alive_before_sigterm']
                if label == 'error':
                    assert report['query_returncode'] == 19
                    assert 'full NVML diagnostic error' in (directory/'health/main-nvidia-smi-q.stderr.log').read_text()
                else:
                    assert report['query_timeout_seconds'] == .02
                    assert time.monotonic()-started < 1
            finally:
                terminate_variant(child)
            cases.append('diagnostic_'+label+'_is_saved_without_retry')
        spec = importlib.util.spec_from_file_location('runtime_benchmark_guard', Path(__file__).with_name('benchmark.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import torch
        graphs = []
        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward
        @torch.compile(backend=backend, fullgraph=True)
        def function(x):
            return (x.sin() + x.cos()).square().sum()
        observer = module.Observer(root)
        scope = {'__name__': 'src.submodules.models.dino.person_detector'}
        exec('class DinoPersonDetector:\n def _load_impl(self):\n  return None\n', scope)
        sys.setprofile(observer.profile)
        try:
            scope['DinoPersonDetector']()._load_impl()
            for i in range(16):
                x = torch.arange(8, dtype=torch.float32) + i
                torch.testing.assert_close(function(x), (x.sin() + x.cos()).square().sum())
        finally:
            sys.setprofile(None)
        assert len(graphs) == 1 and len(observer.calls) == 1
        assert not torch.cuda.is_initialized()
        cases.append('profile_does_not_compile_or_initialize_cuda')
        print(json.dumps({'status': 'passed', 'cases': cases, 'count': len(cases),
                          'cuda_initialized': torch.cuda.is_initialized()}, indent=2))


if __name__ == '__main__':
    main()
