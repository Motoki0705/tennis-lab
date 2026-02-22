import type {
  CellsResponse,
  CourtGeometryResponse,
  SimulateShotRequest,
  SimulateShotResponse,
} from "./types";

// We prefer relative URLs so Next rewrites can proxy to the Python server.
const API_BASE = "/api/blcs";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
  }
  return (await res.json()) as T;
}

export async function apiGetCells(): Promise<CellsResponse> {
  return fetchJson<CellsResponse>(`${API_BASE}/cells`);
}

export async function apiGetCourtGeometry(): Promise<CourtGeometryResponse> {
  return fetchJson<CourtGeometryResponse>(`${API_BASE}/court_geometry`);
}

export async function apiSimulateShot(
  req: SimulateShotRequest
): Promise<SimulateShotResponse> {
  return fetchJson<SimulateShotResponse>(`${API_BASE}/simulate_shot`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
}
