"""Cell geometry helpers for the WebUI API.

We expose cell bounds/centers so the UI can:
- render the same grid as the simulator uses
- map clicks to cell IDs in a deterministic way
"""

from __future__ import annotations

from src.tasks.blcs.generate_dataset.api_server.schemas import (
    CellBounds,
    CellInfo,
    CellsResponse,
    Vec2,
)
from src.tasks.blcs.simulation.cell_manager import CellManager


def build_cells_response(cell_manager: CellManager | None = None) -> CellsResponse:
    cm = cell_manager or CellManager()
    cells: list[CellInfo] = []

    for side in ("near", "far"):
        for cell_id in range(20):
            b = cm.cell_id_to_bounds(cell_id, side)
            center = cm.get_cell_center(cell_id, side)
            cells.append(
                CellInfo(
                    cell_id=cell_id,
                    side=side,  # type: ignore[arg-type]
                    bounds=CellBounds(
                        x_min=b.x_min,
                        x_max=b.x_max,
                        y_min=b.y_min,
                        y_max=b.y_max,
                    ),
                    center=Vec2(x=float(center[0].item()), y=float(center[1].item())),
                )
            )

    return CellsResponse(cells=cells)

