---
description: Stage 2A – Prepare OpenSpec docstrings with human review before coding.
auto_execution_mode: 3
---
<!-- OPENSPEC:START -->
**Guardrails**
- Stay focused on docstring planning and updates—do not modify functional code during this stage.
- Follow Stage 2A guidance in `openspec/AGENTS.md` and the detailed formatting rules in `openspec/docstring_rules.md`.
- Collect human feedback and approval on docstrings before progressing to Stage 2B.

**Steps**
Track these steps as TODOs and complete them one by one.
1. Read `changes/<id>/proposal.md`, `design.md` (if present), and `tasks.md` to confirm scope and acceptance criteria.
2. Inventory every module/class/function you will touch and outline required docstring updates, referencing the governing spec paths.
3. Update module, class, and function docstrings in Google style, ensuring each module `Notes:` cites the relevant spec (e.g., `docs/specs/<capability>.md`).
4. Run `ruff check --select D` (plus optional `pydocstyle`, `darglint`) and resolve issues until docstring lint passes.
5. Share the docstring plan and diff with a human reviewer, capture their feedback, and obtain explicit approval before continuing.
6. Record the approval outcome (e.g., in `tasks.md` or change notes) and proceed to the Stage 2B workflow only after sign-off.

**Reference**
- `openspec/AGENTS.md`
- `openspec/docstring_rules.md`
<!-- OPENSPEC:END -->
