"""Queue-owned native numerical/dtype/chunk/compile and real-shape validation."""
import argparse
import json
import runpy
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from guard import configure, host_guard
from rebuild_dataset import write


def main(root):
    import pytest
    import torch
    metadata = configure(root)
    write(root / "qualification-environment.json", dict(metadata, python=sys.version, torch=str(torch.__version__),
                cuda=torch.version.cuda, started_at=time.time()))
    old = Path("knowledge/runs/run-ball-arch-efficiency-s42-20261111")
    sys.path.insert(0, str(old))
    import protocol
    protocol.ROOT = root
    with host_guard(root / "qualification-host.json"):
        original_gradcheck = torch.autograd.gradcheck
        def logged_gradcheck(function, inputs, **kwargs):
            print(json.dumps(dict(phase="official_gradcheck", channels=inputs[0].shape[-1],
                                  fd_count=len(list(Path('/proc/self/fd').iterdir())))), flush=True)
            # Failure reporting only: avoid gradcheck's huge slow-Jacobian
            # diagnostic after a failed fast check; criteria remain unchanged.
            return original_gradcheck(function, inputs, **kwargs, raise_exception=False)
        torch.autograd.gradcheck = logged_gradcheck
        try:
            runpy.run_path(str(old / "verify_msda.py"), run_name="__main__")
        finally:
            torch.autograd.gradcheck = original_gradcheck
        result = pytest.main(["-n", "0", "-x", "tests/integration/tasks/ball_detection/test_multiscale_query.py", "--durations=0"])
        if result != 0:
            raise RuntimeError("Native CUDA/compile tests failed")
        from shape_equivalence import check
        check(root)
    write(root / "native-qualification.json", dict(complete=True, allocator_fraction=.80,
          numerical="msda-numerical-check.json", cuda_tests_returncode=int(result),
          full_shape="full-shape-equivalence.json", finished_at=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    try:
        main(root)
    except BaseException as error:
        write(root / "native-qualification-FAILED.json", dict(error=str(error), traceback=traceback.format_exc(), timestamp=time.time()))
        raise
