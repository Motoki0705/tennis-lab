"""Review CLI validates its input directory before starting the web server."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.tennis_scene.chat_annotation.scripts import review_ui
from src.utils.configuration.errors import PathContractError


@pytest.mark.parametrize("arguments", [[], ["--root", "relative-output"]])
def test_review_requires_absolute_root(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    create = Mock()
    monkeypatch.setattr(review_ui, "create_app", create)
    monkeypatch.setattr("sys.argv", ["review_ui", *arguments])
    with pytest.raises(SystemExit) as error:
        review_ui.main()
    assert error.value.code == 2
    create.assert_not_called()


@pytest.mark.parametrize("kind", ["filesystem_root", "missing", "file"])
def test_review_rejects_invalid_root_before_starting(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path("/") if kind == "filesystem_root" else tmp_path / kind
    if kind == "file":
        root.write_text("not a directory", encoding="utf-8")
    create, run = Mock(), Mock()
    monkeypatch.setattr(review_ui, "create_app", create)
    monkeypatch.setattr(review_ui.uvicorn, "run", run)
    monkeypatch.setattr("sys.argv", ["review_ui", "--root", str(root)])
    with pytest.raises(PathContractError):
        review_ui.main()
    create.assert_not_called()
    run.assert_not_called()


def test_review_uses_validated_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create, run = Mock(), Mock()
    monkeypatch.setattr(review_ui, "create_app", create)
    monkeypatch.setattr(review_ui.uvicorn, "run", run)
    monkeypatch.setattr(
        "sys.argv", ["review_ui", "--root", str(tmp_path), "--port", "8770"]
    )
    review_ui.main()
    create.assert_called_once_with(tmp_path)
    run.assert_called_once_with(create.return_value, host="127.0.0.1", port=8770)
    assert list(tmp_path.iterdir()) == []
