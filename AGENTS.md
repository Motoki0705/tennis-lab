# AGENTS.md

This file defines implementation-time conventions for this repository.

## 1) Domain overview (what goes where)
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

## 2) Data & outputs policy
- Do not commit large data or model artifacts to git.
- If local data is required, add a short note in docs (or README) describing:
  - expected path under `data/`
  - how to obtain it
  - any licensing/usage constraints
- Use placeholder files or small samples when needed for tests.

### Output directory structure
- All training outputs are organized under `outputs/{module}/{task}/` (e.g., `outputs/plcs/frame/`, `outputs/blcs/single/`)
- Hydra run directories follow the pattern: `${run.output_dir}/${now:%Y-%m-%d_%H-%M-%S}`
  - This creates timestamped subdirectories for each training run
  - Config: set in `hydra.run.dir` within training config files
- Checkpoints are saved at: `{run.output_dir}/logs/version_X/checkpoints/`
  - Example: `outputs/plcs/frame/logs/version_0/checkpoints/last.ckpt`
  - The `version_X` directory is automatically managed by PyTorch Lightning's TensorBoardLogger

## 3) Implementation conventions
- Keep changes within the existing repo structure; do not add new top-level directories.
- Favor explicit, typed interfaces for public functions and module boundaries.
- Use clear error messages for invalid inputs and boundary cases; avoid silent failures.
- Docstrings are required for public APIs and non-trivial logic; keep comments concise and purpose-driven.
- Minimize global state and side effects; keep I/O at the edges of modules.
