"""BLCS dataset generation package.

This package contains the modular implementation of BLCS synthetic dataset generation:
- physics-based shot simulation
- distribution-controlled sampling
- camera sampling + projection
- scene serialization + split/meta writing

The Hydra CLI entrypoint is `src/blcs/scripts/generate_dataset.py`.
"""
