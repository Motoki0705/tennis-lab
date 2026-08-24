# Exploration

- Issue: #786
- Attempt: 6
- Status: COMPLETE
- Frozen issue SHA-256: `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`

## Scope and Issue interpretation

This fresh attempt addresses only Validator RETURN AC-021: the untracked documentation-routing integration test has one hook-scoped mypy `no-any-return` error at `tests/integration/tasks/test_court_coordinate_normalization_documentation.py:32`. The required repair is test-only and must preserve every documentation assertion. No production, configuration, contract, or runtime behavior change is authorized or warranted.

The prior changed-file command was not an adequate canonical replacement for the pre-commit hook: it selected a partial candidate file list, so `src.utils.paths` could be skipped and its exported `PROJECT_ROOT` became `Any`. The closure command must use the hook executable and its exact `mypy --follow-imports=skip` argv while traversing both `src` and `tests`; this includes untracked tests and makes the imported path authority typed.

## Relevant files and symbols

| Path / symbol | Verified role and repair boundary |
|---|---|
| `tests/integration/tasks/test_court_coordinate_normalization_documentation.py::_read` | The sole failing helper. It declares `-> str` and returns `(PROJECT_ROOT / relative_path).read_text(...)`; this is the only test source that needs a narrow explicit local type bridge. |
| `src/utils/paths.py::PROJECT_ROOT` | Declares `PROJECT_ROOT: Path`; it is unchanged from the frozen base. When this module is excluded by `--follow-imports=skip`, importing it from the isolated test produces `Any`; when `src` is a mypy target, it is checked and retains `Path`. |
| `.pre-commit-config.yaml::mypy` | Mandatory hook entry is `./scripts/run_in_repo_venv.sh mypy --follow-imports=skip`, restricted to Python files under `src`, `tests`, or `.spin`; its repository configuration supplies the strict flags. |
| `pyproject.toml::[tool.mypy]` and `[[tool.mypy.overrides]] module = ["tests.*"]` | Global `warn_return_any = true` emits `no-any-return`. Tests relax only untyped-def/decorator ceremony and unreachable/unused-ignore rules; they do not relax return-`Any` checking. |

## Entry points and execution paths

1. The documentation test calls `_read()` for the canonical base README and every routed README/YAML entry point, then checks owner-only prose remains outside routed documents.
2. `_read()` combines imported `PROJECT_ROOT` with a relative `Path` and calls `Path.read_text(encoding="utf-8")`; at runtime the result is a `str`.
3. Under the Validator's changed-file-only invocation, `src/utils/paths.py` is not a target. `--follow-imports=skip` therefore treats `PROJECT_ROOT` as `Any`; `Any / Path` and the following `.read_text()` remain `Any`, which violates the declared `str` return under `warn_return_any`.
4. A `cast(str, ...)` suppresses the isolated `Any` return but fails the broad command with `[redundant-cast]`, because that command traverses `src/utils/paths.py` and sees `PROJECT_ROOT: Path`. A typed local assignment accepts both views: `text: str = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")`; `return text` has no runtime transformation and remains type-correct whether the expression is `Any` or `str`.

## Data, configuration, and interface contracts

| Layer | Contract |
|---|---|
| Documentation helper | `_read(relative_path: Path) -> str` must return decoded UTF-8 document text and keep nonexistent paths/errors fail-loud. |
| Type-only repair | Use `text: str = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")` followed by `return text`; remove the `cast` import. Assignment of `Any` is permitted in isolated changed-file checking, while the same assignment is naturally `str` in broad traversal, so `warn_redundant_casts` cannot reject it. This changes no runtime value. |
| Hook-equivalent broad command | `./scripts/run_in_repo_venv.sh mypy --follow-imports=skip src tests` from the repository root. It preserves the hook executable and full mypy argv, loads `pyproject.toml` strictness, and traverses all `src` and `tests` files—including untracked tests. |
| Non-goal | Do not loosen `warn_return_any`, add an ignore, change `PROJECT_ROOT`, alter the hook filter, or touch production code. |

## Existing tests and fixtures

- `test_court_coordinate_normalization_documentation.py` is untracked and is in the hook's `tests/` scope. Its four tests enforce canonical documentation ownership/routing and must remain unchanged except for the `_read` type bridge.
- `tests/integration/tasks/plcs/test_artifact_publication.py` and `tests/unit/tasks/plcs/generate_dataset/io/test_dataset_io.py` are the other untracked candidate tests. Neither declares a `-> str` helper returning data through an imported skipped module.
- A targeted inventory of `tests/integration/tasks` plus those two neighboring untracked PLCS files found only `_read` in the documentation test as a `-> str` helper around `read_text`; other nearby `read_text` uses feed `json.loads` or lack a declared `str` return boundary.
- Deterministic current evidence: the `cast(str, ...)` variant passes isolated `./scripts/run_in_repo_venv.sh mypy --follow-imports=skip tests/integration/tasks/test_court_coordinate_normalization_documentation.py`, but broad `./scripts/run_in_repo_venv.sh mypy --follow-imports=skip src tests` fails at line 33 with `Redundant cast to "str" [redundant-cast]`. The typed-local assignment has the required static behavior in both modes without an ignore or configuration change.

## Invariants and compatibility constraints

- Preserve the exact UTF-8 read and all documentation-routing assertions; the local assignment must not introduce fallback text, path inference, coercion, or error suppression.
- Retain repository mypy strictness from `pyproject.toml`; the tests override does not authorize `Any` returns.
- Treat the broad `src tests` invocation as the frozen canonical check for this retry, not `pre-commit run --all-files` (which does not reliably enumerate untracked tests) and not a changed-file list that can hide imported type authorities.
- No production code changed during this exploration; `src/utils/paths.py` is verified unchanged from frozen base revision `59e3b166c2d010d5e62be52c2be76d98a94af0e0`.

## Risks and likely impact radius

1. A bare `# type: ignore[no-any-return]` would conceal the returned-value contract; `cast(str, ...)` is also invalid because full traversal rejects it as redundant. The typed local assignment is the narrow bridge accepted in both modes.
2. Changing `PROJECT_ROOT` or global mypy settings would expand the fix beyond the candidate test and could affect unrelated callers.
3. Checking only the repaired test can still reproduce an `Any` imported authority under `--follow-imports=skip`; the required broad command is needed to exercise all source/test targets and untracked tests together.

## Unresolved questions

None. The validator finding has an explicit, test-only repair and a reproducible hook-equivalent broad verification command.

## Evidence table

| Kind | Claim | Evidence |
|---|---|---|
| FACT | AC-021 failed only because of `no-any-return` at documentation test line 32. | `04-validation/validation.md`, focused AC-021 finding. |
| FACT | `PROJECT_ROOT` is explicitly annotated `Path` in unchanged `src/utils/paths.py`. | `src/utils/paths.py:15`; frozen-base content is identical. |
| FACT | The hook invokes `./scripts/run_in_repo_venv.sh mypy --follow-imports=skip` and applies to `src`/`tests` Python files. | `.pre-commit-config.yaml`. |
| FACT | `warn_return_any = true` remains active for `tests.*`. | `pyproject.toml:178-191,254-259`. |
| FACT | The `cast(str, ...)` variant passes isolated mypy but broad `src tests` mypy fails with one `[redundant-cast]` at line 33. | Fresh deterministic executions: isolated PASS (1 source); broad FAIL (1124 sources). |
| FACT | Nearby untracked PLCS tests have no matching declared-string return/read-text boundary. | Focused `rg` inventory of the three untracked candidate tests. |
| INFERENCE | A typed local `text: str` assignment followed by `return text` is the narrowest durable test-only repair. | It preserves the runtime string and fail-loud I/O behavior, accepts imported `Any` in isolated hook-style analysis, and requires no cast when full traversal proves the expression is already `str`. |
| UNKNOWN | No material unresolved contract question remains. | Frozen Issue, focused Validator finding, hook/config, current test, nearby tests, and full candidate status were independently inspected. |
