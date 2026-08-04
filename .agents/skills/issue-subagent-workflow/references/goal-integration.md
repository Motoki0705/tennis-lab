# `/goal` integration

`/goal` is an optional outer persistence loop. `.codex/tasks/issue-<number>/state.toml` remains authoritative for phases, tester cycles, and validator verdicts.

## Why it fits

Codex persists the thread objective and automatically starts another continuation turn while an active goal is idle. Its continuation prompt preserves the full objective and requires evidence-based, requirement-by-requirement completion auditing. This fits both the inner Implementer–Tester loop and the broader Validator RETURN-to-exploration loop.

Primary implementation references in `openai/codex`:

- `codex-rs/ext/goal/src/runtime.rs`: active goals continue when the thread is idle.
- `codex-rs/ext/goal/templates/goals/continuation.md`: the objective remains intact and completion requires authoritative evidence.
- `codex-rs/ext/goal/src/spec.rs`: `update_goal(status="complete")` is allowed only when no required work remains.
- `codex-rs/state/src/model/thread_goal.rs`: goal state is persisted.
- `codex-rs/tui/src/app/thread_goal_actions.rs`: goals require a persisted session.

## Recommended objective

```text
/goal Implement GitHub Issue #<n> using the issue-subagent-workflow. Preserve the full Issue scope. Completion requires: (1) a frozen Issue snapshot with a non-empty normalized acceptance checklist; (2) every AC item implemented; (3) the independent Test Writer records PASS after all relevant tests and checks succeed; (4) issue_validator independently verifies every AC item as PASS; (5) manage_issue_task.py accepts validator PASS and its final check succeeds; and (6) the requested pull request is opened. On tester RETURN, keep the goal active and return to implementation. On validator RETURN, keep the goal active and restart formal exploration. Do not complete the goal from implementation claims, checkbox state, partial tests, or a plausible summary.
```

Use one goal for one Issue.

## Operating rules

- Goal status never advances workflow state. Use `manage_issue_task.py`.
- Tester RETURN is ordinary progress and returns to Implementer.
- Validator RETURN is ordinary progress and returns to Explorer.
- Neither RETURN is, by itself, a blocked goal.
- Goals auto-continue when idle; they are not schedulers. Pause for long human or external waits.
- Do not set a token budget unless explicitly requested.
- Use a saved Codex session.

## Completion gate

Before goal completion, confirm:

1. The Issue and checklist hashes match `state.toml`.
2. `test_verdict = "PASS"` and the recorded `tests.md` test cycle matches state.
3. `validation.md` contains one exact ordered row per AC item and every row is PASS.
4. `manage_issue_task.py verdict ... PASS` succeeds.
5. `manage_issue_task.py check ...` succeeds.
6. Required repository checks have current successful evidence.
7. The requested PR exists and describes both tester and validator gates.

Agent narratives are not substitutes for these gates.
