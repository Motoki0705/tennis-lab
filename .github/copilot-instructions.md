# GitHub Copilot Instructions

This file contains repository-wide instructions for GitHub Copilot. `AGENTS.md` is for Codex; use this file for Copilot-specific behavior.

## Project Context

- This repository develops ML components for tennis analysis.
- Task-specific modules live under `src/tasks/*`.
- Scene-level integration lives under `src/tennis_scene`.
- Shared utilities live under `src/utils`.
- Datasets and dataset-related files live under `data`.
- Generated outputs belong in `outputs/`.
- Use `.venv/bin/python` for project Python commands.

## Issue-Based Approval Workflow

Use this workflow for Copilot implementation tasks that are tied to a GitHub issue.

1. Read the issue and inspect the relevant code before editing.
2. Draft a concrete design comment on the issue before implementation.
3. Immediately after posting the design comment, use the `vscode/askQuestions` tool to ask the user to approve the plan.
4. Do not implement until the user approves through the question response.
5. If the user rejects or changes the plan, post an updated design comment and use `vscode/askQuestions` again.
6. After approval, implement only the approved scope.
7. When implementation reaches the agreed completion criteria, post a result summary comment on the issue.
8. After posting the result summary, wait for user review. If the user requests changes, return to the design-comment plus `vscode/askQuestions` approval loop.
9. Create a PR only when the user explicitly asks for PR creation.

The approval loop is:

```text
design issue comment
-> vscode/askQuestions approval
-> implement or revise design
-> result comment or revised design comment
-> vscode/askQuestions approval when another implementation step is needed
-> repeat until user approves the result
-> create PR only on explicit user command
```

## Approval Comment Guidance

- The first design comment should include requirements, scope, acceptance criteria, and the intended working branch.
- Revision comments can be shorter. They only need to describe the revised direction or options.
- If there are multiple reasonable approaches, list the options in the issue comment and ask the user to choose with `vscode/askQuestions`.
- Result comments should summarize changed files, validation performed, unresolved risks, and whether the completion criteria are met.

## PR Rules

- Do not create a PR automatically after implementation.
- Create a PR only after the user explicitly instructs you to do so.
- Use `Closes #...` for the issue the PR completes.
- Use `References #...` only for additional related issues that should not be closed.
