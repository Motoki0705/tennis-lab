# BLCS generate_dataset WebUI / API Plan

This document is the implementation-time design for the interactive simulator UI.
The goal is to let us *feel* how initial conditions map to trajectories, and to
support controlled experiments like "same from/to cell but different apex/time".

Repository constraint reminder:
- Keep everything under `src/blcs/generate_dataset/` (no new top-level dirs).

## Directory structure (planned)

```
src/blcs/generate_dataset/
  WEBUI_PLAN.md                 # (this file) overall architecture and UX design notes
  api_server/                   # Python HTTP API (local dev server)
    README.md                   # How to run + request/response examples
    # (implementation files; to be added)
    # app.py                    # FastAPI app wiring (routes, middleware)
    # schemas.py                # Pydantic request/response models
    # service.py                # Calls existing simulators; does validation/normalization
    # metrics.py                # Derived metrics: apex, time-to-bounce, net-clearance, etc.
    # cells.py                  # Cell bounds/centers for UI; ID<->bounds helpers
    # __main__.py               # `python -m ...api_server` entrypoint (uvicorn)
  webui/                        # Next.js (React) + TypeScript app (local dev)
    README.md                   # How to run + UX notes + dev workflow
    .gitignore                  # Ignore node_modules/.next/etc (local-only)
    # (implementation files; to be added)
    # package.json
    # next.config.mjs
    # tsconfig.json
    # src/
    #   app/
    #     page.tsx              # Main screen (controls + visualizations)
    #     layout.tsx
    #   components/
    #     CourtPicker.tsx       # 2D top-down court + cell click selection
    #     Trajectory3D.tsx      # 3D view (r3f) + orbit + timeline scrubber
    #     Trajectory2D.tsx      # 2D view (top-down) overlay trajectory + bounce points
    #     ControlsPanel.tsx     # from/to + initial state + physics params
    #     MetricsPanel.tsx      # computed metrics + classification results
    #     PresetPicker.tsx      # Drive/Lob/etc presets
    #   lib/
    #     api.ts                # API client and (optional) proxy endpoints
    #     types.ts              # TS types matching Python schemas
    #     court.ts              # Court geometry, cell mapping helpers for UI rendering
    #     units.ts              # Degree/radian conversions, vector helpers, etc.
```

Notes:
- We intentionally keep Python and Next side-by-side to make iteration on the
  generator/simulator code quick (no extra repo / package split).
- The `webui/` directory is not a Python package (no `__init__.py`) to avoid
  accidental imports.

## UX: what the UI must make obvious

The UI is for intuition-building. It should make these questions answerable
without reading code:
- "If I change elevation by +10 deg, what happens to apex/bounce time/landing cell?"
- "If I keep from/to cell fixed, can I generate both a drive and a lob?"
- "How do drag/magnus/spin change landing vs. a gravity-only approximation?"

### Layout (single screen)
- Left column: Controls
- Right column: Visualization (2D top-down + 3D + metrics)

### Cell selection
We need a clickable court grid (0-19 cell IDs) because cell IDs alone are hard
to interpret. Requirements:
- draw singles court + net line + cell boundaries
- click to pick `from_cell` and `to_cell`
- highlight selected cells and show the sampled/actual bounce point

### "Shot type" knobs (design-first)
We will explicitly surface a knob that corresponds to the planned sampling policy
(policy B):
- apex target (meters) OR time-of-flight target (seconds)

Even before backend supports it, the UI should reserve space for these controls
so the workflow doesn't change when we implement the sampler.

## Python API design (local dev)

Principles:
- Keep it stateless per request (pure function over request params).
- Reuse existing simulator code: `ShotSimulator`, `RallySimulator`,
  `TargetedVelocitySampler`, `CellManager`.
- Validate inputs early and return clear errors (no silent clipping).

### Endpoints (v0)
- `GET /cells`
  - Returns cell bounds/centers for both sides; used to render the court and map
    click events to IDs deterministically.

- `POST /simulate_shot`
  - Runs a single shot simulation and returns:
    - trajectory positions/velocities
    - event indices (net, bounce1, bounce2) + event positions
    - classification (category, to_cell)
    - derived metrics (apex height, time-to-bounce1, net clearance, etc.)

### Request shape (conceptual)
We keep inputs explicit and typed. A request can specify:
- origin:
  - `from_side`: "near" | "far"
  - `from_cell`: int 0..19
- target (optional):
  - `target_mode`: "none" | "cell" | "point"
  - if "cell": `to_cell`: int 0..19
  - if "point": `target_point`: (x,y) in meters (z assumed 0 for bounce target)
- initial state (overrides):
  - position (x,y,z) in meters
  - velocity (vx,vy,vz) in m/s (or speed+angles; API should accept both)
  - spin (wx,wy,wz) in rad/s
- physics:
  - gravity, k_drag, k_magnus, use_drag, use_magnus, e_z, mu, alpha_net
- sim:
  - dt, max_sim_frames, sim_fps, output_fps

## WebUI <-> API integration

Preferred dev ergonomics:
- Next runs on `localhost:3000`
- Python API runs on `localhost:8001`
- Avoid CORS pain by either:
  - Next `rewrites` to proxy `/api/*` to the Python server, OR
  - Next Route Handlers (`src/app/api/.../route.ts`) that proxy requests.

We can decide which approach after we check the repo's current node tooling.

## Implementation steps (suggested)
1. Add `api_server/` with only two endpoints: `/cells`, `/simulate_shot`.
2. Add `webui/` minimal Next app: render court + run button + show trajectory lines.
3. Add 3D view + time slider + metrics.
4. Add presets + JSON export/import to reproduce runs.

