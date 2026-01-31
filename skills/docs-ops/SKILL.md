---
name: docs-ops
description: Documentation governance for tennis-lab. Use when deciding where to document changes, creating or updating README/docs content, or defining documentation structure, conventions, and responsibilities (top-level README vs src/<task>/README vs docs/).
---

# docs-ops

Use this skill to decide document placement and write consistent documentation in this repo.

## Placement rules (decision flow)
- Put **high-level vision and entry points** in the root `README.md`.
- Put **task overviews and example commands** in `src/<task>/README.md`.
- Put **deep details (architecture, specs, parameters, design rationales)** in `docs/`.
- Put **test details** in `docs/tests/`.
- Put **script execution details** in `docs/scripts/<task>/`.
- Put **Docker/runtime environment** details in `docs/docker/`.
- Treat `third_party/` as read-only; add clarifying notes in `docs/` if needed.

## README responsibilities
### Root `README.md`
- Keep it **high-level**: project vision, pipeline overview, module summaries.
- Include **minimal** setup steps and short command examples only.
- Link out to `docs/` and `src/<task>/README.md` for details.

### `src/<task>/README.md`
- Provide **task scope**, **inputs/outputs**, and **typical commands** (short examples).
- Provide **lightweight directory overview** (not exhaustive).
- Redirect architecture/design specifics to `docs/`.

## `docs/` responsibilities
- Capture **detailed architecture, design decisions, configs, and workflows**.
- Keep each document **single-topic**; split when it grows large.
- If adding a new doc category, create an index `docs/<category>/README.md` and link from an appropriate entry point (e.g., `docs/scripts/README.md` or root `README.md`).

## Experiments
- Put experiment-specific notes in `experiments/<name>/README.md`.
- If experiments grow, add a `docs/experiments/README.md` index and link to each experiment README.

## Writing conventions
- Prefer concise sections with clear headings.
- Use `uv run` in command examples.
- Keep README content **overview-level**; push detail into `docs/`.
- Update only the **minimal** set of docs needed for a change.

## Update checklist
- Identify the change scope and choose placement based on the rules above.
- Add/adjust links between README and docs for navigation.
- Keep formatting consistent with neighboring docs.
- Avoid duplicating the same detail in multiple places.
