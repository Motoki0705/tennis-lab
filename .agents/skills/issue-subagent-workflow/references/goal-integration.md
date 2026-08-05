# `/goal` integration

`/goal` is an optional outer persistence loop. `.codex/tasks/issue-<number>/state.toml` remains authoritative for feasibility, phases, preflight, Tester cycles, blockers, and Validator verdicts.

## Recommended objective

```text
/goal Implement GitHub Issue #<n> using issue-subagent-workflow. Completion requires: (1) a frozen non-empty acceptance checklist; (2) feasibility PASS with no unresolved constraint conflict; (3) every AC item implemented; (4) deterministic preflight PASS for the final test cycle; (5) the independent Test Writer records PASS; (6) issue_validator independently records every AC item PASS; (7) manage_issue_task.py accepts Validator PASS and final check; and (8) the requested PR exists. On preflight RETURN, return to implementation without spending a Tester cycle. On the first Tester RETURN, repair and repeat preflight. On the second Tester RETURN, perform return-review before continuing. On Validator RETURN, restart formal exploration. If state becomes blocked, pause the goal rather than auto-continuing or completing it.
```

Use one goal for one Issue.

## Operating rules

- Goal status never advances workflow state; use `manage_issue_task.py`.
- Feasibility BLOCKED or `block` means the goal must pause for an Issue, authority, dependency, or environment change.
- Preflight RETURN is ordinary implementation progress and does not increment `test_cycle`.
- Tester and Validator RETURN are ordinary progress, subject to the repeated-RETURN review gate.
- Do not set a token budget unless explicitly requested.
- Keep the parent thread focused on requirements, state changes, and decisions. Child logs and repeated command output stay outside the parent context.
- Goals auto-continue when idle; they are not schedulers and must not churn against an unresolved blocker.

## Completion gate

Before goal completion, confirm:

1. Issue and checklist hashes match `state.toml`.
2. `feasibility_verdict = "PASS"` or an explicitly normalized legacy task is being completed.
3. The final test cycle has matching `preflight_verdict = "PASS"` and `test_verdict = "PASS"`.
4. No return review or blocker remains unresolved.
5. `validation.md` contains one exact ordered row per AC item and every row is PASS.
6. `manage_issue_task.py verdict ... PASS` succeeds.
7. `manage_issue_task.py check ...` succeeds.
8. Current required repository checks succeed.
9. The requested PR exists and describes feasibility, preflight, Tester, and Validator gates.

Agent narratives and plausible summaries are not substitutes for these gates.
