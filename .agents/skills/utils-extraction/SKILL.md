---
name: utils-extraction
description: Use this skill to decide whether shared or duplicated code in Motoki0705/tennis-lab belongs in src/utils, src/tasks/base, or should remain task-local. This is a placement policy, not a fixed migration workflow.
---

# Utils / base extraction policy

## Purpose

Use this skill when considering whether duplicated or shared code should be
moved into `src/utils` or `src/tasks/base`.

This skill defines the placement criteria. It intentionally does **not** prescribe
a fixed survey, migration, commit, or PR workflow. The calling AI should choose
the concrete implementation approach that best fits the change.

## Required constraints

- Work in a dedicated worktree. Use the `worktree-create` skill when creating it.
- Keep the change focused and reviewable.
- Follow `tests/README.md` for validation. Do not hard-code test paths in this
  skill; use the repo's current testing convention.

## Placement criteria

### Move to `src/utils`

Move code to `src/utils` when it is domain-agnostic and does not depend on a
specific task's types, config shape, labels, model, dataset, or training loop.

Good fits include helpers for paths, IO, device selection, seeding, tensor / numpy
conversion, geometry, rotation math, heatmaps, video, rendering, schemas, and
small numeric utilities.

A useful test: the helper would still make sense in a non-tennis project or in a
new task that does not know about the current task's domain objects.

### Move to `src/tasks/base`

Move code to `src/tasks/base` when it is not task-specific, but it does depend on
shared task-level concepts.

Good fits include common training modules, runners, dataset base classes, scene
layout abstractions, prediction saving, optimizer / scheduler construction, and
other reusable ML-task infrastructure.

A useful test: the helper is too task-aware for `src/utils`, but it is shared by
multiple tasks or defines a reusable task-level abstraction.

### Keep task-local

Keep code inside the task when it is used by only one task or encodes that task's
specific domain assumptions.

Good task-local fits include task-specific losses, model heads, label schemas,
keypoint definitions, dataset-specific preprocessing, config-specific glue, and
one-off experiment code.

## Avoid over-extraction

Do not extract a helper only because it could theoretically be reused later.
Extract when it removes real duplication, clarifies ownership, or gives callers a
more obvious place to import from.

Avoid extracting single-call-site helpers, thin wrappers that only rename another
function, and abstractions whose behavior is only meaningful in one task.

## Import and compatibility policy

When duplicated implementations are consolidated, consumers should import the new
shared implementation directly from `src.utils...` or `src.tasks.base...`.

Do **not** preserve old import paths with delegates or re-export shims as part of
this skill. Update all consumers to the new import path and remove the old copy.

## Behavior preservation

Extraction is an ownership change, not a behavior change. Preserve behavior unless
the user explicitly requested otherwise.

Be especially careful with RNG seeding, dtype and precision, device placement,
argmax tie-breaking, coordinate systems, units, rounding, and serialization
formats.

## Supporting docs

Before adding a new utility, check `src/utils/README.md` when it exists. Keep that
README aligned when the new helper changes how future callers should choose a
utility module.

## Done

- The chosen destination matches the `utils` / `base` / task-local criteria.
- Duplicated implementations are removed rather than left beside the shared one.
- Consumers import directly from the new shared location.
- The change was made in a worktree.
- Validation follows `tests/README.md` and passes under the repo's current test
  convention.
