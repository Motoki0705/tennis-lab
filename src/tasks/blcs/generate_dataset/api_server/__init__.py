"""Local HTTP API for BLCS simulator visualization.

This package is intentionally small and *stateless*:
- The WebUI sends a request describing initial conditions + (optional) target.
- The server runs the existing Python simulators and returns the trajectory/events.

The UI lives next to this under `src/tasks/blcs/generate_dataset/webui/`.
"""

