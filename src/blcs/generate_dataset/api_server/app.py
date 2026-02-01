"""FastAPI app for the BLCS simulator WebUI."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.blcs.generate_dataset.api_server.cells import build_cells_response
from src.blcs.generate_dataset.api_server.court_geometry import (
    build_court_geometry_response,
)
from src.blcs.generate_dataset.api_server.schemas import (
    CellsResponse,
    CourtGeometryResponse,
    SimulateShotRequest,
    SimulateShotResponse,
)
from src.blcs.generate_dataset.api_server.service import simulate_shot
from src.blcs.simulation.cell_manager import CellManager


def create_app() -> FastAPI:
    app = FastAPI(title="BLCS generate_dataset WebUI API", version="0.1.0")

    # Local dev convenience. If we later proxy through Next, this can be tightened.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    cm = CellManager()

    @app.get("/cells", response_model=CellsResponse)
    def get_cells() -> CellsResponse:
        return build_cells_response(cm)

    @app.get("/court_geometry", response_model=CourtGeometryResponse)
    def get_court_geometry() -> CourtGeometryResponse:
        return build_court_geometry_response()

    @app.post("/simulate_shot", response_model=SimulateShotResponse)
    def post_simulate_shot(req: SimulateShotRequest) -> SimulateShotResponse:
        return simulate_shot(req)

    return app


app = create_app()
