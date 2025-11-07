---
description: Stage 2B – Implement approved OpenSpec change with human review.
auto_execution_mode: 3
---
<!-- OPENSPEC:START -->
**Guardrails**
- Keep functional edits tightly scoped to the approved tasks, prioritizing minimal solutions first.
- Follow Stage 2B guidance in `openspec/AGENTS.md` and keep docstrings aligned with code using `openspec/docstring_rules.md` when signatures or behaviors change.
- Secure human feedback on the implementation diff before moving to Stage 2C.

**Steps**
Track these steps as TODOs and complete them one by one.
1. Confirm Stage 2A approval is recorded and revisit `changes/<id>/tasks.md` to plan the implementation order.
2. Implement tasks sequentially, limiting edits to the required files and scenarios outlined in the spec and proposal.
3. Update or add docstrings whenever function signatures, behaviors, or new modules/classes emerge—apply the Google style guidance in `openspec/docstring_rules.md`.
4. Maintain an accurate status log (e.g., in `tasks.md`) but do not mark items complete until the work is verified locally.
5. Document noteworthy decisions or trade-offs directly in the change notes to support review.
6. Present the implementation diff to a human reviewer, gather feedback, and obtain explicit approval before proceeding to testing.

**Reference**
- `openspec/AGENTS.md`
- `openspec/docstring_rules.md`
<!-- OPENSPEC:END -->
