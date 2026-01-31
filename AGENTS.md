# AGENTS.md

This file describes how to work on this repository as a contributor or coding agent.

## 1) Before you start (quick research)
- Skim the repo layout: `src/`, `tests/`, `docs/`, `experiments/`, `data/`, `outputs/`, `third_party/`.
- Confirm the current workflow in `pyproject.toml` (uv/ruff/mypy/pytest) and follow it.

## 2) Domain overview (what goes where)
- `WASB` (`src/wasb`): detect 2D ball position on the image.
- `PLCS` (`src/plcs`): infer 3D player position on the court from 2D skeletons.
- `BLCS` (`src/blcs`): infer 3D ball trajectory on the court from 2D ball positions.
- `third_party/`: external modules (e.g., GVHMR for SMPL/pose). Keep vendor code isolated.
- `data/`: datasets and inputs. Document any local-only data.
- `outputs/`: model checkpoints, artifacts, and generated results.

### Typical data flow
1) Video input -> 2D detections (WASB, court keypoints, skeletons)
2) 2D detections -> court-space 3D (PLCS / BLCS)
3) Optional SMPL/mesh from `third_party/GVHMR` -> fused 3D scene

## 3) Environment & reproducibility rules
- Use `uv sync` to install dependencies.
- Python version is pinned in `.python-version` (3.11). Keep `.venv` local.
- Run scripts via `uv run` to respect locked dependencies.
- For submodules in `third_party/`, initialize with:
  - `git submodule update --init --recursive`
  - If a submodule needs extra setup, document it in the relevant README or docs entry.

## 4) Data & outputs policy
- Do not commit large data or model artifacts to git.
- If local data is required, add a short note in docs (or README) describing:
  - expected path under `data/`
  - how to obtain it
  - any licensing/usage constraints
- Use placeholder files or small samples when needed for tests.

## 5) Testing policy (meaningful tests only)
- Keep the existing structure (no new top-level test directories).
  - Unit tests: `tests/unit` + `@pytest.mark.unit`
  - Integration tests: `tests/unit` + `@pytest.mark.integration`
  - E2E tests: `tests/e2e` + `@pytest.mark.e2e`
- Avoid “tests that only make the CI green.” Prefer:
  - behavior checks (inputs -> expected outputs)
  - boundary/edge cases
  - failure-mode coverage (invalid inputs, missing files, etc.)
- Use existing markers: `unit`, `integration`, `e2e`, `slow`.

## 6) Documentation updates
Before changes, identify impacted docs:
- `README.md` for user-facing pipeline or usage changes.
- `docs/` for deeper references, scripts, and testing notes.
- `experiments/` if experiment configs or outputs change.

Prefer improving structure/clarity, not just appending new text.

## 7) Working checklist
- Confirm target module and data flow impact.
- Update or add tests with meaningful assertions.
- Note any local data requirement and document it.
- Update docs/README if behavior or workflow changes.
- Keep changes within existing repo structure.
