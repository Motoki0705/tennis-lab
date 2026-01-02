"""Simple test for agents module."""


def test_agents_module_imports():
    """Test that agents module can be imported."""
    from src.agents.providers import Provider

    assert Provider.CLAUDE.value == "claude"
    assert Provider.GEMINI.value == "gemini"
    assert Provider.CODEX.value == "codex"
    assert Provider.COPILOT.value == "copilot"


def test_pre_commit_result():
    """Test PreCommitResult dataclass."""
    from src.agents.scripts.pre_commit import PreCommitResult

    result = PreCommitResult(
        status="pass",
        fixed=False,
        summary="test passed",
    )

    json_str = result.to_json()
    assert '"status": "pass"' in json_str
    assert '"fixed": false' in json_str


def test_test_result():
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

    json_str = result.to_json()
    assert '"status": "fail"' in json_str
    assert '"fixed": true' in json_str
    assert '"needs_main": true' in json_str
