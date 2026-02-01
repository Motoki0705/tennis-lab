"""BLCS dataset generation package.

This package contains the modular implementation of BLCS synthetic dataset generation:
- physics-based shot simulation
- distribution-controlled sampling
- camera sampling + projection
- scene serialization + split/meta writing

The Hydra CLI entrypoint is `src/blcs/scripts/generate_dataset.py`.

Developer tooling (planned / design docs live in this folder):
- `src/blcs/generate_dataset/api_server/`: lightweight local HTTP API to run a single-shot
  simulation with user-specified initial conditions and return the trajectory + events.
- `src/blcs/generate_dataset/webui/`: Next.js (React) + TypeScript UI that calls the API
  and visualizes the trajectory in 2D/3D for interactive exploration.
"""
