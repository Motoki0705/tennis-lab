---
name: refactor
description: Use this skill after an experiment has proven its worth and the throwaway code needs to become clean, reusable src/ code — graduating experiments/ or prototype scripts into src/tasks, src/tennis_scene, or src/utils. It defines behavior-preserving refactoring under a green test bar, dead-code removal, and the repo's quality gates (ruff, mypy, script-conventions). Use it for restructuring existing code, not for adding new behavior.
---

# Refactoring (tennis-lab)

## Why this skill exists

This is a research lab: we generate a lot of **experimental code fast** — under
`experiments/`, `experiments_mcmc/`, and ad-hoc `scripts/` — to learn whether an
idea works. That code is meant to be cheap and disposable. The moment an
experiment earns a permanent place, the code that produced it usually does *not*
belong in `src/` as-is. This skill is the bridge: turn proven experiment code
into clean, typed, tested, reusable modules **without changing what it does**.

Refactoring = changing structure while preserving behavior. If you are changing
behavior, that is a feature/fix — use [`tdd`](../tdd/SKILL.md) instead. If you
are only restructuring, the existing behavior is the spec to protect.

## When to use

- An experiment under `experiments/**` proved valuable and its logic should move
  into `src/tasks/<task>`, `src/tennis_scene`, or `src/utils`.
- A `src/` module has accreted dead branches, copy-paste, or leftover experiment
  scaffolding after iteration and needs cleanup.
- You are extracting a shared helper out of two diverging task copies.

Do **not** use this skill to add features, change numerics, or "improve while I'm
here" beyond the agreed scope. Mixing behavior change into a refactor is how
silent regressions enter `src/`.

## The safe refactor loop

1. **Pin behavior first (characterization tests).**
   Before moving anything, capture the current observable contract with tests per
   [`tdd`](../tdd/SKILL.md) — shapes, dtypes, output keys, error cases, a CPU
   smoke step. If the experiment had no tests, this is where they get written.
   These tests are your safety net; they must be green *before* you touch structure.

2. **One behavior-preserving transform at a time.**
   Rename, extract function/class, inline, move module, dedupe — apply a single
   transform, then re-run the pinned tests. Lean on the tools: `ruff check --fix`
   and `ruff format` for mechanical changes.

3. **Keep behavior and structure in separate commits.**
   A refactor commit must not change outputs. If you discover a needed behavior
   change mid-refactor, stop, finish/commit the structural step, then make the
   behavior change as its own tested commit.

4. **Remove dead code as you go.**
   Delete superseded experiment branches, unused parameters, commented-out
   blocks, and now-unreferenced helpers. Confirm with a reference search before
   deleting (`rg "<symbol>"`), and let the tests confirm nothing depended on it.
   Do not keep "just in case" code — git history is the backup.

## Graduating experiment code into `src/`

When promoting code out of `experiments/**`:

- **Placement** (per `AGENTS.md`): task-specific logic -> `src/tasks/<task>/...`;
  cross-task scene logic -> `src/tennis_scene`; reusable helpers -> `src/utils`.
  Keep a task's `data/` / `models/` / `training/` substructure consistent with
  its siblings.
- **Configuration:** replace hardcoded constants and `argparse` with Hydra
  configs under the matching `configs/` directory. Entry-point scripts under
  `src/**/scripts/` must satisfy [`script-conventions`](../script-conventions/SKILL.md)
  (module docstring with Overview -> Usage -> Notes, Hydra config, no `argparse`).
- **Typing:** add type annotations; `src` is checked under strict mypy
  (`disallow_untyped_defs`). New public functions need full signatures.
- **Dependencies:** if a graduated module needs a new package, add it with
  `uv add <pkg>` — never hand-edit dependency tables.
- **Leave the experiment** in place (or delete it) deliberately; don't import
  `src/` code back from `experiments/` to re-create a tangle.

## Definition of done

- The pinned/characterization tests are still green: `.venv/bin/python -m pytest <target>`.
- `.venv/bin/python -m ruff check src tests` is clean (CI enforces this).
- `.venv/bin/python -m mypy src` is clean for the modules touched.
- No dead code remains (no commented-out blocks, unused params, or orphaned
  helpers introduced by the refactor).
- Behavior is unchanged: the diff is structure-only, or any behavior change is
  isolated in its own clearly-labeled, tested commit.
- Promoted scripts comply with [`script-conventions`](../script-conventions/SKILL.md).

## Notes

- Prefer many small, reviewable commits (`refactor:` prefix) over one large diff;
  it makes "structure only, behavior identical" auditable.
- If there are zero tests and writing characterization tests is impractical
  (heavy GPU/data), at minimum add a CPU smoke contract first — never refactor
  `src/` blind.
- Delegate an autonomous cleanup pass to the `refactor-cleaner` subagent
  (`.claude/agents/refactor-cleaner.md`).
- Pair with [`tdd`](../tdd/SKILL.md): TDD adds behavior under new tests; refactor
  changes shape under existing tests. Together they keep `src/` clean as
  experiments churn.
