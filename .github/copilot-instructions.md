# GitHub Copilot Instructions

The single source of truth for every agent in this repository is [`AGENTS.md`](../AGENTS.md) and [`.agents/README.md`](../.agents/README.md) (project overview, development rules, where information lives, GitHub rules, shared memory). Read them before working. This file only adds what is specific to Copilot: an issue-based approval loop that uses `vscode/askQuestions`.

## Issue-Based Approval Workflow

Use this workflow for Copilot implementation tasks tied to a GitHub issue.

1. Read the issue and inspect the relevant code before editing.
2. Post a design comment on the issue (requirements, scope, acceptance criteria, working branch). If there are several reasonable approaches, list them as options.
3. Immediately ask the user to approve the plan with `vscode/askQuestions`. Do not implement until they approve.
4. If the user rejects or changes the plan, post a shorter revised design comment and ask again.
5. After approval, implement only the approved scope.
6. When the agreed completion criteria are met, post a result comment (changed files, validation, unresolved risks, whether the criteria are met) and wait for the user's review. Requested changes go back to step 4.
7. Create a PR only when the user explicitly asks. Use `Closes #...` for the issue it completes and `References #...` for related issues that should stay open.
