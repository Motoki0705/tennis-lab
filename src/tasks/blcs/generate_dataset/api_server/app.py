"""FastAPI app for the BLCS simulator WebUI."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.tasks.blcs.generate_dataset.api_server.cells import build_cells_response
from src.tasks.blcs.generate_dataset.api_server.court_geometry import (
    build_court_geometry_response,
)
from src.tasks.blcs.generate_dataset.api_server.schemas import (
    CellsResponse,
    CourtGeometryResponse,
    SimulateShotRequest,
    SimulateShotResponse,
)
from src.tasks.blcs.generate_dataset.api_server.service import simulate_shot
from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig
from src.tasks.blcs.generate_dataset.simulation.cell_manager import CellManager


def create_app(generator_config: GeneratorConfig) -> FastAPI:
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

    cells_route: Callable[
        [Callable[[], CellsResponse]], Callable[[], CellsResponse]
    ] = app.get("/cells", response_model=CellsResponse)

    @cells_route
    def get_cells() -> CellsResponse:
        return build_cells_response(cm)

    court_geometry_route: Callable[
        [Callable[[], CourtGeometryResponse]], Callable[[], CourtGeometryResponse]
    ] = app.get("/court_geometry", response_model=CourtGeometryResponse)

    @court_geometry_route
    def get_court_geometry() -> CourtGeometryResponse:
        return build_court_geometry_response()

    simulate_shot_route: Callable[
        [Callable[[SimulateShotRequest], SimulateShotResponse]],
        Callable[[SimulateShotRequest], SimulateShotResponse],
    ] = app.post("/simulate_shot", response_model=SimulateShotResponse)

    @simulate_shot_route
    def post_simulate_shot(req: SimulateShotRequest) -> SimulateShotResponse:
        try:
            return simulate_shot(req, generator_config=generator_config)
        except ValueError as e:
            # Map input validation failures to a 4xx for better UX in the WebUI.
            raise HTTPException(status_code=400, detail=str(e)) from e

    return app
