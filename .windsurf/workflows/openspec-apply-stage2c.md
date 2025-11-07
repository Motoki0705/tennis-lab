---
description: Stage 2C – Validate OpenSpec change with focused testing and review.
auto_execution_mode: 3
---
<!-- OPENSPEC:START -->
**Guardrails**
- Limit changes to test code, fixtures, and validation assets; avoid functional edits unless a defect is uncovered.
- Follow Stage 2C guidance in `openspec/AGENTS.md`, and adhere to expectations in `openspec/tests_rules.md` for coverage and reporting.
- Capture command outputs for the human reviewer and obtain approval before archiving or merging.

**Steps**
Track these steps as TODOs and complete them one by one.
1. Review `changes/<id>/tasks.md` to confirm which scenarios require automated coverage and note any outstanding items from Stage 2B feedback.
2. Add or update the necessary unit, integration, and/or end-to-end tests to exercise the implemented behavior.
3. Run the required test suites (e.g., `pytest`, `tox`, project-specific tooling) and record the exact commands and results.
4. Summarize residual risks or skipped tests, if any, and prepare artifacts (logs, coverage data) for human review.
5. Share the test diffs and execution evidence with a human reviewer, incorporate feedback, and secure explicit approval.
6. Once approved, update `tasks.md` so every completed item is marked `- [x]` and reflects the final testing status.

**Reference**
- `openspec/AGENTS.md`
- `openspec/tests_rules.md`
<!-- OPENSPEC:END -->
