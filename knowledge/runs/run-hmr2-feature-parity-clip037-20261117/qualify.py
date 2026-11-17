"""Run the public single-clip main CLI on one full short doubles clip via queue."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import traceback

import numpy as np
import torch

OUT = Path(os.environ['HMR2_ACCEPTANCE_ROOT'])
EXPECTED = os.environ['HMR2_MAIN_COMMIT']


def main() -> None:
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != EXPECTED:
        raise RuntimeError('Main checkout changed after pinning')
    if subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src'], text=True):
        raise RuntimeError('Main source checkout has uncommitted changes')
    OUT.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    repro = Path(os.environ['TENNIS_REPRO_DIR'])
    shutil.copy2(source, repro / 'qualification-wrapper.py')
    (OUT / 'driver.json').write_text(json.dumps({'main_commit': EXPECTED,
        'driver_commit': os.environ['HMR2_ACCEPTANCE_DRIVER_COMMIT'],
        'driver_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}, indent=2))
    torch.cuda.set_per_process_memory_fraction(10 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    code = 1
    try:
        runpy.run_module('src.tennis_scene.scripts.run_pipeline', run_name='__main__', alter_sys=True)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else int(exc.code is not None)
        raise
    except BaseException as exc:
        (OUT / 'failure.json').write_text(json.dumps({'type': type(exc).__name__, 'message': str(exc),
            'traceback': traceback.format_exc()}, indent=2))
        raise
    else:
        code = 0
    finally:
        (OUT / 'process_exit.json').write_text(json.dumps({'exit_code': code, 'main_commit': EXPECTED}, indent=2))


if __name__ == '__main__':
    main()
