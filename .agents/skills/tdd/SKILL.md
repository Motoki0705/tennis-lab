---
name: tdd
description: Use this skill when writing or changing code that will live under src/ (tasks, tennis_scene, utils) or repo tooling, and you want a test-first workflow. It defines the RED -> GREEN -> REFACTOR loop and the tennis-lab testing conventions (contract tests, pytest markers, skip-on-missing-data, CPU smoke, Hydra configs). It does not apply to throwaway experiment code until that code graduates into src/.
---

# Test-Driven Development (tennis-lab)

## Scope

Use this skill when adding or changing code that other code depends on:

- `src/tasks/*` (model/data/training modules), `src/tennis_scene`, `src/utils`
- repo tooling whose contract matters (e.g. `.agents/skills/*/scripts`, knowledge/queue helpers)

This skill is deliberately **pragmatic for a research lab**. Exploratory,
single-use code under `experiments/`, `experiments_mcmc/`, notebooks, and one-off
plotting scripts is **exempt** — write it fast, learn from it, then bring it
under test with [`refactor`](../refactor/SKILL.md) when it graduates into `src/`.
Test the *contract* a piece of code exposes, not the result of a training run.

## The loop: RED -> GREEN -> REFACTOR

1. **RED — write a failing test first.**
   - Express the contract you want (input -> output shape/dtype/keys, raised
     errors, invariants). Run it and *watch it fail for the expected reason*.
   - A test that was never executed in the red state does not count.
   - `git commit` the red test (or stage it) before implementing — it is the spec.

2. **GREEN — write the minimum code to pass.**
   - Implement only enough to make the failing test pass. Re-run the *same*
     target and confirm it is green. Resist adding unrequested behavior.

3. **REFACTOR — clean up under a green bar.**
   - Improve names, remove duplication, tighten types. Re-run the test after
     each change; it must stay green. Then run ruff + mypy (see DoD).
   - For larger cleanups (experiment -> `src/`), hand off to [`refactor`](../refactor/SKILL.md).

Repeat one small contract at a time. Many tiny cycles beat one big one.

## Running tests

Always use the project interpreter (per `AGENTS.md`):

```bash
.venv/bin/python -m pytest tests/tasks/test_dataset_loading_contracts.py -q   # one file
.venv/bin/python -m pytest tests/ -k court -q                                  # by keyword
.venv/bin/python -m pytest -m "unit and not local_data" -q                     # by marker
```

In a git worktree there may be no `.venv`; fall back to `python -m pytest`.
`pytest` is configured in `pyproject.toml` (`[tool.pytest.ini_options]`) with
`testpaths = ["tests"]` and coverage on `src` enabled automatically.

## Conventions (the future habit this skill defines)

### Layout & naming
- Tests live under `tests/`, mirroring `src/` (e.g. `tests/tasks/...`).
- Files: `test_*.py`; classes: `Test*`; functions: `test_*`.
- Name the behavior, not the function: `test_scene_dataset_getitem_returns_fixed_seq_len`.
- Prefer the **contract test** idiom already used in `tests/tasks/test_*_contracts.py`:
  one file groups the public contracts of a task (dataset `__getitem__`,
  model `forward` shapes, runner construction/smoke).

### Markers (declared in `pyproject.toml`)
Apply exactly one tier marker plus any capability markers:

| Marker | Use for |
| --- | --- |
| `unit` | pure functions/classes, no GPU, no disk, fast |
| `integration` | several components together (datamodule + model, config + runner) |
| `e2e` | a full workflow end-to-end |
| `slow` | anything that is not quick; lets others `-m "not slow"` |
| `local_data` | needs large local datasets/checkpoints — **not runnable in CI** |
| `cuda` | needs a GPU; must skip cleanly when unavailable |

```python
import pytest
pytestmark = pytest.mark.local_data        # whole module
@pytest.mark.unit                          # single test
```

### Skip, don't fail, when the environment can't run it
Tests must stay green on a clean checkout with no datasets and no GPU. Guard
external prerequisites and `pytest.skip` with a clear reason — copy the existing
helper style:

```python
def _require_paths(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"local dataset assets are missing: {', '.join(map(str, missing))}")

if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)
```

### Models / training: assert contracts on CPU, tiny inputs
Don't assert learned metrics. Assert shapes, dtypes, output keys, and that a
single step runs. Force CPU and minimal sizes:

- training smoke overrides: `run.gpus=0`, `data.batch_size=1`, tiny dims / 1 step.
- set `torch.manual_seed(0)` for any test that touches randomness.
- build Hydra configs the way the smoke tests do, and reset global state:

```python
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="config", overrides=["run.gpus=0", "data.batch_size=1"])
```

### Coverage
`--cov=src` runs by default and reports term/html/xml. Treat coverage as a
*lens for finding untested contracts*, not a hard gate — there is no blanket
percentage to chase. **CI today runs only `ruff check src tests`, not pytest**,
so a test that is green only on your machine is not protected by CI: keep new
tests CPU- and data-free (skip otherwise) so any teammate can reproduce green.

## What to test vs. skip

- **Do test:** dataset `__getitem__` contracts, model `forward` output shapes/keys,
  datamodule wiring, runner construction + 1-step smoke, data transforms/augmentations,
  pure `src/utils` helpers, knowledge/queue tooling behavior.
- **Don't gate on:** notebooks, `experiments/**` exploration, visual/plot output,
  exact numeric training results, anything requiring network or private assets.

## Definition of done

- The new/changed contract has a test that was seen failing, then passing.
- `.venv/bin/python -m pytest <target>` is green (or skips for missing data/GPU
  with a clear reason — never silently passes on an empty collection).
- `.venv/bin/python -m ruff check src tests` is clean (CI enforces this).
- `.venv/bin/python -m mypy src` is clean for typed code you touched.
- Tests run on CPU without local datasets, or `pytest.skip` when they cannot.

## Notes

- New scripts under `src/**/scripts/` or `experiments/**/scripts/` must also
  satisfy [`script-conventions`](../script-conventions/SKILL.md) (docstring + Hydra, no argparse).
- Delegate the full test-first cycle to the `tdd-guide` subagent
  (`.claude/agents/tdd-guide.md`) when you want it driven autonomously.
- Prefer a few sharp contract tests over many brittle ones; a test that breaks on
  every refactor is testing implementation, not contract.
