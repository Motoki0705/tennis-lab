---
name: experiment-result-pr
description: "Use this skill when a tennis-lab training or experiment run has finished and the user wants the result registered in knowledge-control, reported back to the GitHub issue, worked from a git worktree, and submitted through a PR. It wraps the recurring workflow: inspect the issue next step, locate finished training-queue runs, promote them to knowledge/, write findings, validate, comment on the issue, commit, push, and create or update the PR."
---

# Experiment Result PR

Use this for the repeated request pattern: "the training from the issue's next step finished; register it in knowledge-control, post the result to the issue, work in a worktree, and submit the artifacts as a PR."

## Required Skills

Load these skills before acting:

- `knowledge-control` for run registration, node writing, curves, and validation.
- `gh-issue` for reading/updating the GitHub issue.
- `worktree-create` when a new worktree is needed; if an appropriate issue worktree already exists, use it.
- `gh-pr-create` when creating a new PR. If a PR for the same issue/branch already exists, update that PR instead.

If scripts under `src/**/scripts/` or `experiments/**/scripts/` are changed, also load `script-conventions`.

## Workflow

1. Read the issue with comments:

```bash
gh issue view <issue> --repo Motoki0705/tennis-lab --comments --json number,title,body,comments,labels,state,url
```

Identify the latest requested "next step" and the expected queue job names or configs.

2. Work in a worktree.

- Prefer an existing clean worktree for the same issue/PR; if its branch is already merged, branch fresh off `origin/main` inside it (don't commit onto the merged branch).
- Otherwise create one using the `worktree-create` naming convention, then rename the branch to the repo convention if useful.
- Do not edit from the main checkout. A linked worktree's `.training_queue` / `.venv` are symlinks to the main checkout and are **not** gitignored there — never `git add -A`; stage `knowledge/` and other intended files explicitly.

3. Locate finished queue artifacts.

If the worktree lacks `.training_queue` / `.venv`, symlink them to the main checkout first. Then list the issue's finished runs **by run.json issue field** (not job name — one queue batch can mix issues):

```bash
.venv/bin/python .agents/skills/experiment-result-pr/scripts/discover_runs.py --issue <issue>
```

It marks status / predictions / already-registered, prints a ready-to-paste `kg_register.py` line per run, and flags jobs named `i<issue>…` that actually belong to another issue. Register only the runs it lists for this issue.

4. Register every finished run.

Run the lines `discover_runs.py` printed (`--provider` is taken from run.json; add `--id <run-id>` only to override the default `run-<name>`):

```bash
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_register.py <job-name> --issue <issue> --provider <from run.json>
```

If the queue artifact is outside the worktree, pass `--repro-dir <path>`. Registration records `artifacts.output_dir`, so `kg_curves` resolves the run's TensorBoard directly instead of scanning every event file.

5. Write knowledge findings.

- One node per run.
- Fill `parents`, `relations`, and `tags`.
- Write the fixed run-node `## 考察 / Findings` sections from `knowledge-control`.
- Update or create a group node when the run belongs to an ablation set.
- Use real metrics only; separate observations from hypotheses.

6. Generate curves and validate:

```bash
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_curves.py <run-id>
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_validate.py
```

7. Comment on the issue.

Post a concise Japanese result summary with:

- run ids and PR URL or branch if PR is not created yet,
- a compact metrics table,
- the conclusion and changed interpretation,
- the next recommended experiment.

Use `gh issue comment <issue> --repo Motoki0705/tennis-lab --body-file <tmpfile>`.

8. Submit through PR.

- If an open PR already tracks the issue work, commit and push to that PR branch.
- Otherwise create a PR using `gh-pr-create`.
- Fill the repository PR template in Japanese.
- Mention `Closes #<issue>` only when the PR completes the issue; otherwise use `References #<issue>`.

## Validation Checklist

- `git status --short --branch` is clean except intended changes before committing (the `.training_queue` symlink must stay unstaged).
- `kg_validate.py` exits 0.
- Every new run has `knowledge/runs/<run-id>/{run.json,repro.sh,metrics.json,pred_test.npz}`.
- Every new run has a findings body and meaningful graph links.
- The issue comment matches the metrics in node frontmatter.
- The PR branch is pushed and the PR URL is reported to the user.
