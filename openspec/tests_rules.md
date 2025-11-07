# OpenSpec Testing Rules

Use these rules during Stage 2C (Testing & Verification) to design, implement, and review automated tests for OpenSpec changes. Align every test addition with the approved proposal/tasks and governing specs referenced in `openspec/AGENTS.md`.

## Directory Structure

Organize files under `tests/` using the following layout (create subdirectories as needed):

```
tests/
├── conftest.py             # Session-wide fixtures and pytest config
├── fixtures/               # Reusable fixture modules (data builders, factories)
├── data/                   # Static sample assets (<=1 MB per file, committed to git)
├── helpers/                # Test-only utilities that wrap production code safely
├── unit/                   # Fast, isolated tests of single modules/functions/classes
│   └── <feature>/test_*.py
├── integration/            # Tests that exercise multiple components together
│   └── <capability>/test_*.py
└── e2e/                    # High-level workflows matching spec scenarios
    └── test_*.py
```

* Place feature-specific fixtures alongside the tests that consume them (`tests/unit/<feature>/conftest.py`, etc.).
* Keep top-level `conftest.py` focused on environment setup or fixtures shared project-wide.
* Prefer module-level factory helpers over ad-hoc mocks scattered across tests.

## Pytest Configuration Defaults

Settings defined in `pyproject.toml` apply automatically when running `pytest` (including via CI):

* **Discovery Paths:** `tests/` (from `testpaths`). Python files named `test_*.py` or `*_test.py`, classes starting with `Test`, and functions starting with `test_` are collected.
* **Default Options (`addopts`):**
  * `--verbose`
  * `--cov=tracknet`
  * `--cov-report=term-missing`
  * `--cov-report=html:htmlcov`
  * `--cov-report=xml`
  * `--cov-fail-under=75`
  * `--strict-markers`
  * `--disable-warnings`
* **Coverage Policy:** Overall coverage must remain ≥75% (CI will fail below this threshold). HTML reports are written to `htmlcov/` for local inspection.
* **Registered Markers:**
  * `unit`, `integration`, `e2e`, `slow`. Strict markers mean new marks must be added to `pyproject.toml` before use.
* **Warnings:** Deprecation and pending deprecation warnings are ignored by default via `filterwarnings`.

> Always run tests through `uv run` to ensure these settings and dependencies are respected (see command examples below).

## Test Categories & Expectations

### Unit Tests (`tests/unit/`)
* **Scope:** Single module/class/function with external dependencies mocked.
* **Speed:** Must run in <1s each; avoid filesystem/network unless explicitly required.
* **Assertions:** Cover success, failure, and edge cases defined in spec scenarios.
* **Naming:** `test_<subject>_<behavior>()` inside files named `test_<module>.py`.

### Integration Tests (`tests/integration/`)
* **Scope:** Multiple modules collaborating (e.g., model + data loader + config parsing).
* **Environment:** Use realistic configuration/fixtures; minimal mocking (only for non-deterministic or external services).
* **Data:** Use assets in `tests/data/` or generated via fixtures; document assumptions in fixture docstrings/comments.
* **Reporting:** Log key intermediate outputs when debugging pipelines (sample predictions, metrics).

### End-to-End Tests (`tests/e2e/`)
* **Scope:** Reproduce end-user workflows or spec scenarios from input through observable output.
* **Stability:** Prefer deterministic seeds and bounded data volume to keep runtime predictable.
* **Evidence:** Capture command invocations and summarize results in the Stage 2C hand-off (pass/fail, metrics, artifacts).
* **Frequency:** Only add when the scenario isn’t sufficiently covered by integration/unit tiers; keep runtime manageable.

## Spec Alignment & Traceability
* Map each test module to the governing spec using module-level comments/docstrings referencing `docs/specs/<capability>.md` and requirement/scenario IDs when available.
* In Stage 2C summaries, explicitly list which scenarios gained coverage and link to test names.
* If a spec scenario cannot be automated, document the manual verification plan and rationale.

## Fixtures, Helpers, and Utilities
* Shared factories/data builders live under `tests/fixtures/` (Python modules) or `tests/data/` (static files). Name helpers by capability or domain (e.g., `video.py`, `annotation.py`).
* Keep `tests/helpers/` pure-Python and side-effect free. If a helper grows beyond test support, move it into production code with appropriate specs.
* Use `pytest` fixtures for lifecycle management; prefer context managers or helpers only when fixtures would add unnecessary complexity.

## Test Quality Gates
* Maintain Arrange-Act-Assert structure and avoid multiple unrelated assertions per test.
* Use `pytest.mark.integration` / `.e2e` / `.slow` decorators to toggle heavier suites; strict markers are enforced by default.
* Enforce deterministic seeds (`torch.manual_seed`, `numpy.random.seed`) whenever randomness is involved.
* For GPU-dependent tests, guard with `pytest.importorskip` or custom markers (`skip_if_no_gpu`).

## Execution Checklist
During Stage 2C, capture the commands you execute and their outcomes (replace markers as needed for targeted runs):

```
uv run pytest
uv run pytest -m "unit"
uv run pytest -m "integration and not slow"
uv run pytest -m "e2e" --maxfail=1
```

* Include additional commands for linting/type-checking if they validate test artifacts (e.g., `ruff`, `mypy`).
* Attach logs/coverage reports when they substantiate behavior or bug fixes.

## Review & Sign-off
* Share diffs and test outputs with the human reviewer, highlighting new/modified coverage.
* Address review feedback promptly; re-run impacted suites and update the Stage 2C checklist before marking tasks complete.
* After approval, ensure all relevant tasks in `changes/<id>/tasks.md` are marked `- [x]` and reference the executed test evidence.
