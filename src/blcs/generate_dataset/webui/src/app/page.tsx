"use client";

import { useEffect, useMemo, useState } from "react";

import { apiGetCells, apiGetCourtGeometry, apiSimulateShot } from "../lib/api";
import type {
  CellInfo,
  CourtGeometryResponse,
  Side,
  SimulateShotRequest,
  SimulateShotResponse,
  TargetMode,
} from "../lib/types";
import { ControlsDrawer } from "../components/ControlsDrawer";
import { MetricsPanel } from "../components/MetricsPanel";
import { Trajectory3D } from "../components/Trajectory3D";

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function deg2rad(d: number) {
  return (d * Math.PI) / 180;
}

function computeVelocityFromAngles(params: {
  speed: number;
  azimuthDeg: number;
  elevationDeg: number;
  fromSide: Side;
}) {
  const az = deg2rad(params.azimuthDeg);
  const el = deg2rad(params.elevationDeg);
  const baseDir = params.fromSide === "near" ? 1 : -1;

  const cosEl = Math.cos(el);
  const sinEl = Math.sin(el);
  const sinAz = Math.sin(az);
  const cosAz = Math.cos(az);

  const vx = params.speed * cosEl * sinAz;
  const vy = params.speed * cosEl * cosAz * baseDir;
  const vz = params.speed * sinEl;
  return { vx, vy, vz };
}

export default function Page() {
  const [drawerOpen, setDrawerOpen] = useState(true);

  const [cells, setCells] = useState<CellInfo[]>([]);
  const [court, setCourt] = useState<CourtGeometryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [fromSide, setFromSide] = useState<Side>("near");
  const [fromCell, setFromCell] = useState<number>(0);
  const [targetMode, setTargetMode] = useState<TargetMode>("none");
  const [toCell, setToCell] = useState<number>(0);

  // Start position is represented as "cell-relative offsets" for slider control.
  const [offsetX, setOffsetX] = useState(0.5);
  const [offsetY, setOffsetY] = useState(0.5);
  const [z0, setZ0] = useState(1.0);

  // Launch parameters (sliders).
  const [speed, setSpeed] = useState(25.0);
  const [azimuthDeg, setAzimuthDeg] = useState(0.0);
  const [elevationDeg, setElevationDeg] = useState(18.0);

  // Spin (sliders).
  const [spinX, setSpinX] = useState(0);
  const [spinY, setSpinY] = useState(-60);
  const [spinZ, setSpinZ] = useState(0);

  // Physics toggles.
  const [useDrag, setUseDrag] = useState(true);
  const [useMagnus, setUseMagnus] = useState(true);

  // Camera controls.
  const [cameraMode, setCameraMode] = useState<"orbit" | "fps">("orbit");
  const [fpsMoveSpeed, setFpsMoveSpeed] = useState(8);

  // Simulation state.
  const [running, setRunning] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [simResult, setSimResult] = useState<SimulateShotResponse | null>(null);

  const targetSide: Side = fromSide === "near" ? "far" : "near";

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    Promise.all([apiGetCells(), apiGetCourtGeometry()])
      .then(([cellsRes, courtRes]) => {
        if (!mounted) return;
        setCells(cellsRes.cells);
        setCourt(courtRes);
        setLoadError(null);
      })
      .catch((e) => {
        if (!mounted) return;
        setLoadError(String(e));
      })
      .finally(() => {
        if (!mounted) return;
        setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const fromCellInfo = useMemo(
    () => cells.find((c) => c.side === fromSide && c.cell_id === fromCell) ?? null,
    [cells, fromSide, fromCell]
  );

  const startPos = useMemo(() => {
    if (!fromCellInfo) return null;
    const b = fromCellInfo.bounds;
    const x = lerp(b.x_min, b.x_max, offsetX);
    const y = lerp(b.y_min, b.y_max, offsetY);
    return { x, y, z: z0 };
  }, [fromCellInfo, offsetX, offsetY, z0]);

  const launchVel = useMemo(
    () => computeVelocityFromAngles({ speed, azimuthDeg, elevationDeg, fromSide }),
    [speed, azimuthDeg, elevationDeg, fromSide]
  );

  async function onRun() {
    setRunning(true);
    setSimError(null);
    try {
      const req: SimulateShotRequest = {
        from_side: fromSide,
        from_cell: fromCell,
        target_mode: targetMode,
        to_cell: targetMode === "cell" ? toCell : undefined,
        shot: {
          position: startPos ?? undefined,
          velocity: { x: launchVel.vx, y: launchVel.vy, z: launchVel.vz },
          spin: { x: spinX, y: spinY, z: spinZ },
        },
        physics: { use_drag: useDrag, use_magnus: useMagnus },
        sim: {},
      };
      const res = await apiSimulateShot(req);
      setSimResult(res);
    } catch (e) {
      setSimError(String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div style={{ height: "100vh", width: "100vw", background: "#000" }}>
      <Trajectory3D
        positions={simResult?.positions ?? null}
        court={court}
        cells={cells}
        fromSide={fromSide}
        fromCell={fromCell}
        toCell={targetMode === "cell" ? toCell : null}
        targetSide={targetSide}
        cameraMode={cameraMode}
        fpsMoveSpeed={fpsMoveSpeed}
        bounce1Pos={simResult?.events.bounce1_pos ?? null}
        bounce2Pos={simResult?.events.bounce2_pos ?? null}
        netPos={simResult?.events.net_pos ?? null}
      />

      {/* Minimal HUD */}
      <button
        onClick={() => setDrawerOpen(true)}
        style={{
          position: "fixed",
          top: 16,
          left: 16,
          zIndex: 9,
          display: drawerOpen ? "none" : "block",
          border: "1px solid rgba(255,255,255,0.22)",
          background: "rgba(10,10,10,0.6)",
          color: "#fff",
          borderRadius: 12,
          padding: "10px 12px",
          cursor: "pointer",
          backdropFilter: "blur(6px)",
        }}
      >
        Open Controls
      </button>

      <ControlsDrawer
        open={drawerOpen}
        setOpen={setDrawerOpen}
        fromSide={fromSide}
        setFromSide={(s) => {
          setFromSide(s);
          setFromCell(0);
        }}
        fromCell={fromCell}
        setFromCell={setFromCell}
        targetMode={targetMode}
        setTargetMode={setTargetMode}
        toCell={toCell}
        setToCell={setToCell}
        offsetX={offsetX}
        setOffsetX={setOffsetX}
        offsetY={offsetY}
        setOffsetY={setOffsetY}
        z0={z0}
        setZ0={setZ0}
        speed={speed}
        setSpeed={setSpeed}
        azimuthDeg={azimuthDeg}
        setAzimuthDeg={setAzimuthDeg}
        elevationDeg={elevationDeg}
        setElevationDeg={setElevationDeg}
        spinX={spinX}
        setSpinX={setSpinX}
        spinY={spinY}
        setSpinY={setSpinY}
        spinZ={spinZ}
        setSpinZ={setSpinZ}
        useDrag={useDrag}
        setUseDrag={setUseDrag}
        useMagnus={useMagnus}
        setUseMagnus={setUseMagnus}
        cameraMode={cameraMode}
        setCameraMode={setCameraMode}
        fpsMoveSpeed={fpsMoveSpeed}
        setFpsMoveSpeed={setFpsMoveSpeed}
        running={running}
        onRun={onRun}
      />

      <div style={{ position: "fixed", right: 16, top: 16, zIndex: 9, width: 360 }}>
        <MetricsPanel result={simResult} />
        {loading ? (
          <div style={hudBox()}>Loading geometry...</div>
        ) : loadError ? (
          <div style={{ ...hudBox(), borderColor: "rgba(255,0,0,0.35)" }}>
            <div style={{ color: "#fff", fontSize: 12 }}>Load error</div>
            <div style={{ color: "#fff", fontSize: 12, opacity: 0.8 }}>{loadError}</div>
          </div>
        ) : null}
        {simError ? (
          <div style={{ ...hudBox(), borderColor: "rgba(255,0,0,0.35)" }}>
            <div style={{ color: "#fff", fontSize: 12 }}>Sim error</div>
            <div style={{ color: "#fff", fontSize: 12, opacity: 0.8 }}>{simError}</div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function hudBox(): React.CSSProperties {
  return {
    marginTop: 12,
    background: "rgba(10,10,10,0.78)",
    border: "1px solid rgba(255,255,255,0.14)",
    borderRadius: 14,
    padding: 12,
    backdropFilter: "blur(6px)",
  };
}

