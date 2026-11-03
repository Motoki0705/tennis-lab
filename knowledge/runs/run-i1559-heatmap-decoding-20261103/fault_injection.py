"""Prove the rotated-Gaussian assertion detects a missing XY cross term."""

import json

import pytest

import src.tasks.ball_detection.inference.heatmap_decoding as decoding

NODE = "tests/unit/tasks/ball_detection/inference/test_heatmap_decoding.py::test_known_centres_include_rotation_anisotropy_and_saturation[quadratic_log-3-False-True-2e-05]"


def main():
    original = decoding._design

    def without_cross_term(*args):
        xy, inverse, rank = original(*args)
        inverse = inverse.copy()
        inverse[4] = 0
        return xy, inverse, rank

    try:
        decoding._design = without_cross_term
        injected = int(pytest.main(["-n", "0", "-q", NODE]))
    finally:
        decoding._design = original
    restored = int(pytest.main(["-n", "0", "-q", NODE]))
    receipt = {"node": NODE, "mutation": "zero fitted XY cross coefficient", "injected_exit": injected, "restored_exit": restored}
    print(json.dumps(receipt, sort_keys=True))
    if injected != 1 or restored != 0:
        raise RuntimeError("Fault injection did not produce the expected assertion failure and recovery")


if __name__ == "__main__":
    main()
