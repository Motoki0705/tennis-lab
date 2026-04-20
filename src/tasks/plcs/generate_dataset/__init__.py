"""PLCS dataset generation package.

This package contains the implementation of synthetic PLCS scene generation:
- motion sampling (AMASS/SMPL-H)
- player placement on court
- camera sampling + projection into 2D keypoints
- scene serialization (npy + json)

The Hydra CLI entrypoint is `src/tasks/plcs/scripts/generate_dataset.py`.
"""

