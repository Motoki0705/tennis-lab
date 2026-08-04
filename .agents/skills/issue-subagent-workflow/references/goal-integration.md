# `/goal` integration

`/goal` is an optional outer persistence loop for this workflow. The workflow state machine in `.codex/tasks/issue-<number>/state.toml` remains the authoritative phase and verdict state.

## Why it fits

The Codex goal runtime persists a thread objective and, while the goal is active, starts another continuation turn whenever the thread becomes idle. The continuation prompt requires evidence-based progress, preserves the full objective across turns, and permits completion only after a requirement-by-requirement audit. This matches the workflow's repeated exploration, implementation, independent testing, and validator RETURN loop.

Primary implementation references in `openai/codex`:

- `codex-rs/ext/goal/src/runtime.rs`: active goals continue automatically when the thread is idle.
- `codex-rs/ext/goal/templates/goals/continuation.md`: the objective remains intact across turns and completion requires authoritative evidence for every requirement.
- `codex-rs/ext/goal/src/spec.rs`: `update_goal(status="complete")` is allowed only when the objective is achieved and no required work remains.
- `codex-rs/state/src/model/thread_goal.rs`: persisted statuses include active, paused, blocked, usage-limited, budget-limited, and complete.
- `codex-rs/tui/src/app/thread_goal_actions.rs`: goals require a persisted session rather than an ephemeral thread.

## Recommended objective

Start the saved Codex session in the repository, then set one goal for one Issue:

```text
/goal Implement GitHub Issue #<n> using the issue-subagent-workflow. Preserve the full Issue scope. Completion requires: (1) a frozen Issue snapshot with a non-empty normalized acceptance checklist; (2) every AC item implemented; (3) independent tests and required repository checks completed; (4) issue_validator independently verifies every AC item as PASS; (5) manage_issue_task.py accepts verdict PASS and state.toml records status="complete" and verdict="PASS"; and (6) the requested pull request is opened. On validator RETURN, keep the goal active and restart formal exploration. Do not mark the goal complete from implementation claims, checkbox state, partial tests, or a plausible summary.
```

The objective should identify the exact Issue. Do not combine unrelated Issues into one goal.

## Operating rules

- Goal status does not advance workflow phases. Use `manage_issue_task.py` for all phase transitions and verdicts.
- A validator RETURN is normal progress, not a blocked goal. Restart exploration and keep the goal active.
- Call goal completion only after the helper accepts PASS and all other deliverables in the objective exist.
- The source `[x]` or `[ ]` state in the GitHub Issue is not completion evidence. Only validator evidence and the accepted workflow verdict count.
- Do not use a token budget unless the user explicitly requests one. Budget exhaustion is not success.
- Goals auto-continue immediately when idle; they are not schedulers. When progress depends on a long external wait, such as a human action or delayed external service, pause the goal with `/goal pause` and resume it later with `/goal resume` rather than creating a polling loop.
- Goals require a saved session. If the session is ephemeral, start `codex` normally or resume a persisted session before using `/goal`.

## Completion gate

Before allowing the goal to become complete, the parent must confirm all of the following from current state:

1. `issue.md` contains the normalized AC checklist and its hash matches `state.toml`.
2. `validation.md` contains exactly one row for every AC ID, in order, and every verdict is PASS.
3. `manage_issue_task.py verdict ... PASS` succeeds.
4. `manage_issue_task.py check ...` succeeds.
5. Required tests and checks have current successful evidence.
6. The requested PR exists and describes the checklist-based validation.

A narrative statement from any agent is not a substitute for these gates.
