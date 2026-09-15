"""Serve the fixed, locally packaged Three.js modules used by scene viewers."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

SHARED = Path(__file__).parent / "shared"
ASSETS = frozenset(
    {
        "three.module.js",
        "three.core.js",
        "OrbitControls.js",
        "scene3d.mjs",
        "LICENSE-three.txt",
    }
)


def mount_scene_assets(app: FastAPI) -> None:
    def asset(name: str) -> FileResponse:
        if name not in ASSETS:
            raise HTTPException(404)
        return FileResponse(
            SHARED / name,
            media_type="text/plain" if name.endswith(".txt") else "text/javascript",
            headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
        )

    app.add_api_route("/shared/{name:path}", asset, methods=["GET"])
