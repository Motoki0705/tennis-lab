"""Task inspection extensions preserve the shared routes and asset boundary."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.tasks.base.visualization.review.web import create_review_app


class Service:
    def catalog(self) -> dict[str, Any]:
        return {"task": "test"}

    def scenes(self, form: str) -> dict[str, Any]:
        return {"form": form, "scenes": []}

    def scene(
        self, form: str, scene_id: str, revision: str | None = None
    ) -> dict[str, Any]:
        return {"scene_id": scene_id}

    def buffer(self, form: str, scene_id: str, revision: str | None = None) -> bytes:
        return b"data"


def test_default_shared_assets_and_routes_are_preserved() -> None:
    client = TestClient(create_review_app(Service(), title="Test", task="test"))
    assert "/static/app.js" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/catalog").json() == {"task": "test"}
    assert client.get("/static/undeclared.js").status_code == 404


def test_extensions_load_before_shared_module_and_do_not_replace_it(
    tmp_path: Path,
) -> None:
    script = tmp_path / "inspection.js"
    script.write_text("window.inspection = true;", encoding="utf-8")
    style = tmp_path / "inspection.css"
    style.write_text(".inspection { color: red; }", encoding="utf-8")
    client = TestClient(
        create_review_app(
            Service(),
            title="Test",
            task="test",
            extra_static={"inspection.js": script, "inspection.css": style},
            extra_modules=("inspection.js",),
            extra_stylesheets=("inspection.css",),
        )
    )
    html = client.get("/").text
    assert html.index("/static/inspection.js") < html.index("/static/app.js")
    assert "/static/style.css" in html and "/static/inspection.css" in html
    assert client.get("/static/inspection.js").text == script.read_text()
    assert client.get("/static/app.js").text != script.read_text()
    assert client.get("/api/scene/buffer?form=single&scene=scene").content == b"data"


@pytest.mark.parametrize(
    "name", ["app.js", "../inspection.js", 'evil".js', "index.html"]
)
def test_reserved_or_unsafe_extension_names_are_rejected(
    tmp_path: Path, name: str
) -> None:
    asset = tmp_path / "extension.js"
    asset.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid or reserved"):
        create_review_app(
            Service(), title="Test", task="test", extra_static={name: asset}
        )


def test_undeclared_extension_is_rejected() -> None:
    with pytest.raises(ValueError, match="not declared"):
        create_review_app(
            Service(), title="Test", task="test", extra_modules=("missing.js",)
        )
