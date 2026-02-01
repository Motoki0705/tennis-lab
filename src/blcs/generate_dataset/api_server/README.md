# BLCS generate_dataset API server (planned)

This directory is reserved for a lightweight local HTTP API that exposes the
existing BLCS simulators (`ShotSimulator`, `TargetedVelocitySampler`, etc.)
to the WebUI under `../webui/`.

Why an API server?
- We want the UI to be "thin": render + controls only.
- Physics/simulation stays in Python (single source of truth).
- Iteration is fast: tweak simulator code and refresh the UI.

## Planned module layout

```
src/blcs/generate_dataset/api_server/
  README.md        # This document
  app.py           # FastAPI app + routes (GET /cells, POST /simulate_shot)
  schemas.py       # Pydantic models (request/response)
  service.py       # Simulation orchestration (validation -> simulate -> metrics)
  metrics.py       # Derived metrics (apex, time-to-bounce, net-clearance, etc.)
  cells.py         # Cell bounds/centers for UI + click mapping helpers
  __main__.py      # Entry point for local dev server (uvicorn)
```

## API surface (v0)

- `GET /cells`
  - Returns the court grid definition (bounds/centers) so the UI can:
    - render the 0..19 cell grid consistently
    - map click coordinates to cell IDs

- `POST /simulate_shot`
  - Runs a single shot simulation.
  - Input supports:
    - from_side/from_cell (required)
    - optional target mode (none/cell/point)
    - explicit initial state overrides (pos/vel/spin)
    - physics + sim parameters
  - Output returns:
    - trajectory (positions/velocities)
    - event frames (net/bounce1/bounce2)
    - classification (category, to_cell)
    - derived metrics

## How to run (local dev)

From the repo root:

```bash
UV_CACHE_DIR=/tmp/uv_cache uv run python -m src.blcs.generate_dataset.api_server --reload --port 8001
```

Quick smoke test:

```bash
curl -s http://127.0.0.1:8001/cells | head
curl -s -X POST http://127.0.0.1:8001/simulate_shot \\
  -H 'content-type: application/json' \\
  -d '{\"from_side\":\"near\",\"from_cell\":0,\"target_mode\":\"none\"}' | head
```

## Dev notes

- Keep endpoints stateless (no server-side session).
- Return clear error messages for out-of-range inputs.
- Avoid silently clipping user inputs; if we clamp, report the clamped value.
