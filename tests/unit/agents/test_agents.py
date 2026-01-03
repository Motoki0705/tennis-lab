"""Simple test for agents module."""


def test_agents_module_imports() -> None:
    """Test that agents module can be imported."""
    from src.agents.providers import Provider

    assert Provider.CLAUDE.value == "claude"
    assert Provider.GEMINI.value == "gemini"
    assert Provider.CODEX.value == "codex"
    assert Provider.COPILOT.value == "copilot"


def test_pre_commit_result() -> None:
    """Test PreCommitResult dataclass."""
    from src.agents.scripts.pre_commit import PreCommitResult

    result = PreCommitResult(
        status="pass",
        fixed=False,
        summary="test passed",
    )

    output = result.format_output()
    assert "STATUS: pass" in output
    assert "FIXED: false" in output
    assert "SUMMARY: test passed" in output


def test_test_result() -> None:
    """Test TestResult dataclass."""
    from src.agents.scripts.test import TestResult

    result = TestResult(
        status="fail",
        fixed=True,
        files_touched=["test.py"],
        summary="tests failed",
        needs_main=True,
        message_for_main="Fix required",
    )

    output = result.format_output()
    assert "STATUS: fail" in output
    assert "FIXED: true" in output
    assert "NEEDS_MAIN: true" in output
    assert "FILES_TOUCHED: test.py" in output


def test_exclude_paths() -> None:
    """Test that .venv/ and other paths are excluded."""
    from src.agents.scripts.pre_commit import is_excluded_path

    # Should be excluded
    assert is_excluded_path(".venv/lib/python3.11/site-packages/foo.py")
    assert is_excluded_path("venv/lib/foo.py")
    assert is_excluded_path("__pycache__/foo.pyc")
    assert is_excluded_path(".git/hooks/pre-commit")
    assert is_excluded_path("node_modules/package/index.js")
    assert is_excluded_path(".mypy_cache/3.11/foo.py")
    assert is_excluded_path("src.egg-info/PKG-INFO")

    # Should NOT be excluded
    assert not is_excluded_path("src/agents/scripts/pre_commit.py")
    assert not is_excluded_path("tests/unit/test_foo.py")
    assert not is_excluded_path("pyproject.toml")
