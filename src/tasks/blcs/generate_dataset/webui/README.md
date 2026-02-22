# BLCS generate_dataset WebUI

Next.js (React) + TypeScript WebUI to visualize BLCS ball trajectories generated
by the Python simulator under `src/blcs/generate_dataset/api_server/`.

Goal:
- Make the mapping from initial conditions -> trajectory intuitive.
- Support cell-based workflows: select `from_cell` / `to_cell`, then explore how
  changing parameters changes bounce position, apex height, and events.

## Layout

```
src/blcs/generate_dataset/webui/
  README.md
  .gitignore
  package.json
  next.config.mjs
  tsconfig.json
  next-env.d.ts
  src/
    app/
      layout.tsx
      page.tsx              # main: 3D scene + slider controls + metrics HUD
    components/
      ControlsDrawer.tsx    # slider-based controls (initial conditions, camera mode)
      Court3D.tsx           # court mesh + lines + net + cell highlights
      FpsControls.tsx       # pointer lock + WASD movement
      Trajectory3D.tsx      # r3f scene wrapper (court + trajectory + event markers)
      MetricsPanel.tsx      # apex/time/net-clearance + events
    lib/
      api.ts                # API client (calls Python server)
      types.ts              # request/response types aligned with Python schemas
```

## UX / Features
- 3D-only view renders a recognizable tennis court (lines + net) and the ball trajectory.
- Camera navigation:
  - `orbit` mode: mouse orbit/zoom/pan
  - `fps` mode: click to lock pointer + WASD to move, ESC to unlock
- Slider-based controls:
  - `from_side`, `from_cell`, optional `to_cell` (target mode `cell`)
  - cell-relative start offsets (`offset_x`, `offset_y`) + `z0`
  - `speed`, `azimuth_deg`, `elevation_deg`
  - `spin (wx, wy, wz)`
  - `use_drag`, `use_magnus`
- Metrics HUD:
  - category / classified `to_cell`
  - apex height, time to bounce, net clearance
  - event frames (net/bounce1/bounce2)

## How to run (local dev)

1) Start Python API server (repo root):

```bash
UV_CACHE_DIR=/tmp/uv_cache uv run --group webui -m src.blcs.generate_dataset.api_server --reload --port 8001
```

2) Start Next.js dev server (separate terminal):

```bash
cd src/blcs/generate_dataset/webui
npm_config_cache=/tmp/npm_cache npm install
npm run dev
```

Then open `http://localhost:3000`.

Notes:
- The UI calls `/api/blcs/*` which Next rewrites to `http://127.0.0.1:8001/*`.
- If you want a different API base, set `BLCS_API_BASE` when running `npm run dev`.

## API endpoints used
- `GET /cells` (via `/api/blcs/cells`): cell bounds/centers (used for highlighting)
- `GET /court_geometry` (via `/api/blcs/court_geometry`): CourtKP20 + segments (court lines)
- `POST /simulate_shot` (via `/api/blcs/simulate_shot`): run a single-shot simulation

## Git ignore notes (repo-wide ignore workarounds)
This repo's top-level `.gitignore` ignores `*.json` and `lib/` directories globally.
To keep WebUI configs tracked, `src/blcs/generate_dataset/webui/.gitignore` re-includes:
- `package.json`, `package-lock.json`, `tsconfig.json`
- `src/lib/**`
