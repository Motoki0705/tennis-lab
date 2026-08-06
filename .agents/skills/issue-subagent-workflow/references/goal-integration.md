# `/goal` integration

Read this file only when `/goal` is active. The goal is an outer persistence loop; `state.toml` remains the workflow authority.

A suitable objective is:

```text
Implement GitHub Issue #<n> with issue-subagent-workflow. Complete only after feasibility PASS, every AC is implemented, production preflight PASS, independent Tester PASS, final candidate seal PASS, Issue-only Validator PASS, capture-pr binds the real paginated PR diff and final-head checks to the validated candidate, all required remote checks PASS, finalize-pr succeeds, and the final whole-task check returns ok. Pause on BLOCKED. Continue after preflight, Tester, seal, or Validator RETURN according to state.
```

Operating rules:

- one goal per Issue;
- goal status never mutates workflow state;
- BLOCKED pauses the goal instead of churning;
- `status = "validated"` is not goal completion;
- do not complete before `finalize-pr` and final `check`;
- keep parent context to requirements, decisions, state changes, and compact handoffs;
- do not set a token budget unless explicitly requested.
