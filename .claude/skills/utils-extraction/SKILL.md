---
name: utils-extraction
description: Use this skill to run a periodic or on-demand "extract shared code" campaign in Motoki0705/tennis-lab — survey the repo for duplicated (WET) or util-izable helpers and consolidate them into src/utils (domain-agnostic) or src/tasks/base (cross-task), behaviour-preservingly, to shrink the repo and raise maintainability. Covers the sonnet-subagent survey, the utils-vs-base classification rubric, behaviour-preserving migration, the validation gates (incl. the mypy caveat), and the worktree+PR flow.
---

# Utils / base extraction campaign

## When to use

- A **periodic maintenance pass** to compress the repo and reduce duplication.
- When you notice a generic helper copy-pasted across task modules.
- A request to "util化 / base化 / DRY / dedupe / extract shared / consolidate".

This skill is the **campaign methodology** — the find-and-consolidate loop. The
catalogue of what already lives in `src/utils` and the "where do I add a new
helper" rules are in **`src/utils/README.md`**; read that first. This skill drives
the survey and migration around it. The reuse policy is also stated in
`AGENTS.md` / `CLAUDE.md` ("Code reuse and shared utilities").

## Targets: where extracted code goes

| Destination | Take it here when | Examples |
|-------------|-------------------|----------|
| `src/utils/<module>` | the helper has **no dependency on any task's domain types** (generic path/device/seed/IO/tensor/geometry/heatmap/render/schema/video math) | `resolve_project_path`, `to_numpy`, `angular_error`, `axis_angle_to_rotation_matrix` |
| `src/tasks/base/...` | it depends on shared **task** concepts (scene layout, Lightning module, runner) **and is reused by ≥2 tasks** | `BaseLightningModule` optimizer/scheduler + test-prediction saving, shared scene dataset/runner logic |
| leave in the task | used by exactly **one** task and tied to its domain | task-specific losses, model heads |

Rule of thumb: no task-domain types → `src/utils`. Depends on shared task
concepts but reused by 2+ tasks → `src/tasks/base`. One task only → leave it.
**Do not over-extract** — a single-call-site helper or a one-task abstraction
adds indirection without removing duplication.

## Workflow

### 1. Worktree

Work in a dedicated worktree (use the **worktree-create** skill). Base it on
`origin/main` so the survey reflects the merged state, not a stale branch.

### 2. Survey — delegate to sonnet subagents (required)

The repo exploration **must** be done by sonnet subagents (read-only `Explore`
or `general-purpose`), not inline — this keeps the main context focused and lets
the census fan out. Launch several **in parallel**, each scoped to one slice:

- one per `src/tasks/{ball_detection,blcs,court_detection,plcs}`
- one for `src/tasks/base` + `src/tennis_scene`
- one for `src/**/scripts/` and `experiments/`

Give each subagent the same brief: find **(a)** helpers duplicated across modules
(same/near-same body in 2+ places) and **(b)** module-local functions with **no
task-domain dependency** that belong in `src/utils`. Each finding must report:
`file:line`, the signature, a one-line "why generic / where duplicated", and the
nearest existing `src/utils` (or `src/tasks/base`) module it fits.

Seed greps for the subagents (starting points — they then read for semantic
duplicates, not just textual matches):

```bash
# private helpers that look generic
grep -rnE 'def _?(resolve|ensure|load|save|to_numpy|normalize|denormalize|seed|make_.*rng|clamp|rotation|axis_angle|wrapped|angular)_?' src/tasks src/tennis_scene
# near-duplicate bodies (ImageNet stats, device autoselect, json dump, mmap load)
grep -rn '0.485\|0.456\|0.406\|cuda.*is_available\|json.dump\|mmap_mode' src/tasks src/tennis_scene
```

Consolidate the findings into a short **ranked** list — rank by impact
(`#copies × LOC`) against risk (numeric/RNG-sensitive code is higher risk). For a
large campaign, optionally write the survey to `docs/refactoring/`; for a small
one, keep it in the PR body.

### 3. Classify each candidate

Apply the target table above. Drop anything that turns out to be task-specific or
single-call-site. Keep the batch small enough to review in one PR.

### 4. Migrate behaviour-preservingly

For each accepted candidate:

1. **Copy the body verbatim** into the closest existing `src/utils` / `base`
   module (create a new module only when nothing fits). Behaviour must not
   change — be especially careful with **RNG seeding** (worker-id offsets, base
   seed), **numeric dtype/precision** (bf16/fp16 upcasts), and **argmax
   tie-breaking**. When in doubt, keep the exact arithmetic.
2. **Replace every WET copy with an import.** Preserve public import paths: turn
   the old location into a thin **delegate** (`return shared_fn(...)`) or a
   **re-export shim** (`from src.utils.x import f`). Never leave both copies.
3. **Export** it from the module's `__all__` and the package `__init__`; add a
   row to the `src/utils/README.md` "I need to…" table.
4. **Add a unit test** in `tests/test_utils_extraction.py` asserting the new
   util's behaviour (and, where cheap, that the old import path still resolves to
   it).

### 5. Validate (gates)

Run with `.venv/bin/python`:

- **Import smoke** — import every touched module (catches broken delegations).
- `pytest tests/test_utils_extraction.py`
- **Training-smoke contracts** for each affected task:
  `pytest tests/tasks/test_training_smoke_contracts.py -k "<task>"`
- **ruff** clean.
- **mypy caveat** — pre-commit runs `mypy --follow-imports=skip`. With that flag,
  a cross-module delegation `return imported_fn(...)` resolves to `Any` →
  `no-any-return`, **pervasively, and pre-existing throughout the repo**. Do
  **not** chase these with `# type: ignore[no-any-return]`: mypy batches files and
  then flags the ignores as *unused*. Keep new `src/utils` modules mypy-clean in
  isolation (annotate their own locals), and commit the migration with
  `SKIP=mypy git commit ...`, which matches how these files are necessarily
  committed in this repo. Keep ruff green regardless.

### 6. Scope discipline

- One focused, reviewable PR per campaign.
- If a migration drags in heavy **pre-existing** type debt or
  **uninstalled-dependency** modules (e.g. GVHMR / `hmr4d` in the tennis_scene
  pipeline), **revert that item and defer it** — note the deferral in the PR
  body. Don't let one stubborn item bloat the diff.

### 7. PR

Open the PR with the **gh-pr-create** skill. In the body list each extraction
(what moved, from → to, how many copies removed), the validation evidence, and
any deferred items.

## Environment gotchas

- A fresh worktree lacks the gitignored `data/` dirs, so training-smoke fails
  with `Scene directory not found: data/<task>`. **Symlink** `data/plcs`,
  `data/blcs`, `data/court` from the main tree to run the smoke tests, then
  **remove the symlinks before committing**.
- Use `.venv/bin/python` for every command; manage deps with `uv`.
- This repo keeps skills in **both** `.agents/skills/` and `.claude/skills/`.
  When you add or edit a skill, write it to **both** trees, byte-identical.

## Definition of done

- Every accepted candidate is migrated: no remaining WET copy, public import
  paths preserved, exported, README-listed, unit-tested.
- Gates green: ruff, `tests/test_utils_extraction.py`, import smoke, and
  training-smoke for affected tasks. (mypy intentionally skipped per repo
  reality; new utils are mypy-clean in isolation.)
- Deferred items explicitly listed in the PR.
- Work done in a worktree; one focused PR opened.
