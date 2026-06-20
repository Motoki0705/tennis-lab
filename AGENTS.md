# AGENTS.md

## Project overview

- This repository develops ML components for tennis analysis.
- Task-specific modules live under `src/tasks/*` (for example `ball_detection`).
- Task integration and scene-level analysis live under `src/tennis_scene`.
- Shared reusable modules live under `src/utils`.
- Datasets and dataset-related files live under `data`.
- Generated outputs belong in `outputs/`.

## Python environment

- Use `.venv/bin/python` as the Python runtime for all project commands and scripts.
- When adding dependencies, use `uv add <package>`.
- When removing dependencies, use `uv remove <package>`.
- Do not edit dependency definitions manually when `uv` can manage them.

## Experiment logging & reproducibility (issue #533)

Checkpoints are **not** the unit of record — they are large and disposable. The
record is a reproducibility bundle plus the model's test-split predictions.

- **Always launch training via the `training-queue` skill** (`training_queue.sh
  add ...`), never `python -m ...train` directly. The queue auto-captures the
  git state (commit/branch/remote/uncommitted patch) and the exact command, and
  generates a one-shot `repro.sh`.
- **AI callers MUST pass `--provider` and `--session`** (and `--issue` when
  applicable) so the run is attributable in the knowledge graph. Humans running
  by hand may omit them. To find your own session id, see
  `.agents/skills/training-queue/reference/{claude,codex,gemini}.md`.
- The **lightning module saves test-split inference** to
  `$TENNIS_REPRO_DIR/predictions/pred_test.npz` (+ `metrics.json`) following
  `data/{plcs,blcs}/test.txt`. New metrics are recomputed from this, so the
  checkpoint is not needed to evaluate.
- After a run finishes, **register it with the `knowledge-control` skill**
  (`kg_register.py`), which promotes the bundle + predictions into git-tracked
  `knowledge/runs/<run-id>/` and scaffolds a node linking log / issue / session.
- **Deleting checkpoints is manual**:
  `.agents/skills/training-queue/scripts/prune_ckpts.py` (backfill predictions →
  verify → delete). It never deletes a checkpoint without a verified npz.
