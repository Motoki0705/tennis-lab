---
name: tasks-extension
description: Extend `src/tasks` by first reading existing task implementations such as BLCS and PLCS, then deriving conventions from them instead of relying on a hard-coded template.
---

# Tasks Extension Workflow

## Scope
Use this skill when adding a new task under `src/tasks`, or when restructuring an existing task to align with the repository's task conventions.

## Working principle
- Start with a subagent-led codebase survey before making structural decisions.
- Do not start from a fixed directory template written in this skill.
- Derive the extension strategy by reading existing task implementations.
- Prefer `src/tasks/blcs` as the primary reference.
- Read other tasks such as `src/tasks/plcs` when BLCS alone does not explain a design choice.

## Subagent-first workflow
1. Spawn the project-scoped `tasks_codebase_explorer` subagent defined at `.codex/agents/tasks-codebase-explorer.toml`.
2. Ask that subagent to survey the target task and the closest existing task references before implementation starts.
3. Wait for a concise summary with concrete file references, structural guidance, and open questions.
4. Use the subagent's survey to guide local implementation work instead of redoing the same exploration on the main thread.
5. If the user asked only for analysis, return the survey findings rather than editing code.

Suggested delegation prompt:
```text
Survey the codebase for extending or restructuring <target task>.
Follow the tasks-extension inspection order.
Return the closest reference task, required shared abstractions, pipeline boundary considerations, and the highest-signal files to inspect next.
Do not make code changes.
```

## Minimum inspection order
1. Read the target task's current files, if the task already exists.
2. Read `src/tasks/blcs/README.md`.
3. Read `src/tasks/blcs` entry points under `configs/` and `scripts/`.
4. Read `src/tasks/blcs` implementation layers in this order:
   - `training/`
   - `inference/`
   - `data/`
   - `visualization/`
   - `generate_dataset/` when relevant
5. Read `src/tasks/plcs/README.md` and corresponding modules if multiview, sequence, or dataset-generation patterns need a second reference.
6. Confirm shared abstractions in:
   - `src/tasks/base`
   - `src/utils`
7. If the task will be consumed by the integrated pipeline, confirm the boundary with:
   - `src/tennis_scene/README.md`
   - `src/tennis_scene/pipeline/components/*.py`

## What to confirm before editing
- What responsibility the target task owns.
- Which existing task is the closest structural reference.
- Which parts should stay task-local.
- Which parts should be shared through `src/tasks/base` or `src/utils`.
- Whether the task must expose a predictor contract consumed by `src/tennis_scene`.

## Notes
- This skill is intentionally procedural rather than prescriptive.
- Task-specific structure should be inferred from repository precedents, not copied from a static checklist here.
