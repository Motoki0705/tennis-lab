---
name: script-conventions
description: Use this skill when a script under src/**/scripts/ or experiments/**/scripts/ is modified or newly created. This skill defines and verifies the required script docstring and configuration conventions for repository scripts.
---

# Script Conventions

## When to use

Use this skill when adding or editing any file under `src/**/scripts/` or `experiments/**/scripts/`.

## What this skill enforces

- Every target script must start with a module docstring.
- The module docstring must follow this format:

```python
"""
<overview sentence or short paragraph>

Usage:
    python path/to/script.py ...

Notes:
    - note 1
    - note 2
"""
```

- The docstring must contain all three sections in order:
  - Overview
  - `Usage:`
  - `Notes:`
- Scripts must use Hydra for configuration handling.
- Configuration must come from the corresponding `configs/` directory for that script.
- `argparse` must not be used.

## Required review flow

1. Identify the changed or newly created files under `src/**/scripts/` or `experiments/**/scripts/`.
2. Check that each file has a module docstring at the top of the file.
3. Verify that the docstring uses the required `Overview -> Usage -> Notes` structure.
4. Verify that the script uses Hydra-based configuration loading.
5. Verify that configuration is sourced from the corresponding `configs/` path for that script.
6. Verify that `argparse` is not imported or used.

## Definition of done

- Every changed target script is checked against all rules above.
- Any violations are reported with file paths and concrete reasons.
- If all checks pass, the result clearly states that the scripts comply with the convention.

## Notes

- Keep the skill scoped to scripts under `src/` and `experiments/`; do not apply it to unrelated modules.
- Prefer Hydra patterns already used in the repository over inventing a new configuration style.
- If a script cannot use `python path/to/script.py ...` literally because of the project runner, still include a concrete `Usage:` example that shows the intended invocation shape.
