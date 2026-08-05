# Completion hardening

These rules address failure modes observed in long, multi-attempt Issue workflows. They supplement the state-machine, document, and spawn contracts.

## User-directed topology overrides

Parallel implementation is a default latency optimization, not an acceptance criterion.

- An explicit user request to use one Implementer, sequential implementation, or another bounded topology is compliant.
- Record the request and the chosen topology in `plan.md` under implementation ownership. Do not describe the topology itself as a workflow violation.
- A single Implementer may own several units sequentially. Preserve explicit file ownership, ordered handoffs, and one artifact integrator.
- A topology override does not silently broaden Issue scope, weaken deterministic preflight, reduce required evidence, or remove the independent Validator.
- The independent Test Writer remains the default. When the user explicitly combines implementation and test authorship, record the reduced independence and compensate with canonical deterministic checks plus an independent Validator.

## Canonical command authority

The plan is the source of truth for repository checks. Record each canonical command with its exact arguments, working directory, relevant environment, and authority.

Before a Test Writer emits RETURN because a command failed:

1. Compare the executed invocation with the canonical invocation byte-for-byte in all material arguments.
2. If they differ, run the canonical command.
3. If the canonical command passes, report the non-canonical failure as a diagnostic command mismatch and keep the same test cycle. Do not emit Tester RETURN and do not increment `test_return_count`.
4. A broader diagnostic command may force RETURN only when it independently proves a specific AC defect, not merely because it uses stricter or different tool options.

The parent rejects a Tester RETURN whose only evidence comes from a non-canonical invocation. Correct the command and rerun the same independent test cycle.

## Artifact pre-submit gate

Every formal artifact owner runs the targeted format check before returning control:

```bash
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py \
  artifact-check .codex/tasks/issue-<number> \
  <feasibility|exploration|plan|implementation|preflight|tests|validation>
```

This gate checks the current attempt, placeholders, required headings, exact AC mappings, cycle metadata, hashes, and standalone verdicts that apply to the artifact. A format failure is repaired by the artifact owner before the parent applies a phase or verdict transition.

Do not defer heading or table repair until after Validator PASS.

## Candidate revision binding

Preflight, Test Writer, and Validator artifacts identify the inspected candidate revision. For a dirty worktree, use a deterministic content fingerprint in addition to HEAD.

- Any production or test content change invalidates downstream PASS evidence.
- History-only packaging, squash, or recommit is allowed after validation only when the final content fingerprint is identical to the validated candidate.
- Before creating the PR, compare the final remote diff or tree fingerprint with the validated candidate and rerun affected gates when they differ.

## Fail-closed completion

`verdict PASS` must validate the complete artifact set before writing `status = "complete"`.

- A malformed earlier artifact, stale attempt number, missing heading, mismatched cycle, or invalid AC matrix leaves the state unchanged in validation.
- The state file is never the first place where completion is recorded. Artifact validation precedes state mutation.
- Run the final `check` before PR creation and once more after any content-changing packaging step.

## Context rollover

Authoritative artifacts, not a long agent conversation, carry state between attempts.

- After a Validator RETURN, prefer a fresh Explorer and a fresh Implementer session for the new attempt, even when the user selected a single-Implementer topology.
- Reuse an agent only when retained local context has clear value and remains bounded. Do not keep one thread across repeated attempts merely to avoid a concise handoff.
- Retry messages contain only affected AC IDs, the new failure bundle, ownership, and artifact paths.
- If a child thread has accumulated broad logs or several cycles, roll over to a fresh thread and rely on the artifacts.

## Final packaging checklist

Before PR creation, verify:

1. targeted `artifact-check` succeeds for every formal artifact;
2. preflight, Tester, and Validator PASS refer to the same candidate content;
3. `manage_issue_task.py verdict ... PASS` succeeds without intermediate state corruption;
4. `manage_issue_task.py check ...` returns `ok`;
5. the complete paginated PR diff stays inside the allowed scope;
6. required remote checks succeed on the final PR head.
