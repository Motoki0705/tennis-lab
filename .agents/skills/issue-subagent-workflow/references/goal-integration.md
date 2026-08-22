# `/goal` integration

Read only with `/goal`. It is an outer persistence loop; `state.toml` remains authoritative.

```text
Implement GitHub Issue #<n> with issue-subagent-workflow. Complete only after feasibility PASS, every AC is implemented, independent Preflight Reviewer PASS, independent Test Writer PASS, independent Seal Reviewer PASS, Issue-only Validator PASS, capture-pr binds the real paginated PR diff and final-head checks to the validated candidate, all required remote checks PASS, finalize-pr succeeds, and the final whole-task check returns ok. Pause on BLOCKED. Continue after Preflight Reviewer, Test Writer, Seal Reviewer, or Validator RETURN according to state.
```

One goal per Issue. Goal status never mutates workflow state. BLOCKED pauses rather than churns. `status = "validated"` is not completion; require `finalize-pr` and final `check`. Keep parent context to requirements, decisions, state changes, and compact handoffs. Set no token budget unless requested.
