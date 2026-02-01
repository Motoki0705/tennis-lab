"use client";

import { useEffect, useMemo, useState } from "react";

import { CourtPicker } from "../components/CourtPicker";
import { MetricsPanel } from "../components/MetricsPanel";
import { ControlsPanel } from "../components/ControlsPanel";
import { Trajectory2D } from "../components/Trajectory2D";
import { Trajectory3D } from "../components/Trajectory3D";
import type {
  CellInfo,
  Side,
  SimulateShotRequest,
  SimulateShotResponse,
  TargetMode,
} from "../lib/types";
import { apiGetCells, apiSimulateShot } from "../lib/api";

export default function Page() {
  const [cells, setCells] = useState<CellInfo[]>([]);
  const [loadingCells, setLoadingCells] = useState(true);
  const [cellsError, setCellsError] = useState<string | null>(null);

  const [fromSide, setFromSide] = useState<Side>("near");
  const [fromCell, setFromCell] = useState<number>(0);
  const [targetMode, setTargetMode] = useState<TargetMode>("none");
  const [toCell, setToCell] = useState<number>(0);

  const [posX, setPosX] = useState<string>("");
  const [posY, setPosY] = useState<string>("");
  const [posZ, setPosZ] = useState<string>("");

  const [velX, setVelX] = useState<string>("");
  const [velY, setVelY] = useState<string>("");
  const [velZ, setVelZ] = useState<string>("");

  const [spinX, setSpinX] = useState<string>("");
  const [spinY, setSpinY] = useState<string>("");
  const [spinZ, setSpinZ] = useState<string>("");

  const [useDrag, setUseDrag] = useState(true);
  const [useMagnus, setUseMagnus] = useState(true);

  const [running, setRunning] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [simResult, setSimResult] = useState<SimulateShotResponse | null>(null);

  const targetSide: Side = fromSide === "near" ? "far" : "near";

  useEffect(() => {
    let mounted = true;
    setLoadingCells(true);
    apiGetCells()
      .then((res) => {
        if (!mounted) return;
        setCells(res.cells);
        setCellsError(null);
      })
      .catch((e) => {
        if (!mounted) return;
        setCellsError(String(e));
      })
      .finally(() => {
        if (!mounted) return;
        setLoadingCells(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const fromCells = useMemo(
    () => cells.filter((c) => c.side === fromSide),
    [cells, fromSide]
  );
  const toCells = useMemo(
    () => cells.filter((c) => c.side === targetSide),
    [cells, targetSide]
  );

  function parseMaybeFloat(s: string): number | undefined {
    const t = s.trim();
    if (!t) return undefined;
    const v = Number(t);
    return Number.isFinite(v) ? v : undefined;
  }

  async function onRun() {
    setRunning(true);
    setSimError(null);
    setSimResult(null);
    try {
      const req: SimulateShotRequest = {
        from_side: fromSide,
        from_cell: fromCell,
        target_mode: targetMode,
        to_cell: targetMode === "cell" ? toCell : undefined,
        shot: {
          position:
            parseMaybeFloat(posX) !== undefined &&
            parseMaybeFloat(posY) !== undefined &&
            parseMaybeFloat(posZ) !== undefined
              ? {
                  x: parseMaybeFloat(posX)!,
                  y: parseMaybeFloat(posY)!,
                  z: parseMaybeFloat(posZ)!,
                }
              : undefined,
          velocity:
            parseMaybeFloat(velX) !== undefined &&
            parseMaybeFloat(velY) !== undefined &&
            parseMaybeFloat(velZ) !== undefined
              ? {
                  x: parseMaybeFloat(velX)!,
                  y: parseMaybeFloat(velY)!,
                  z: parseMaybeFloat(velZ)!,
                }
              : undefined,
          spin:
            parseMaybeFloat(spinX) !== undefined &&
            parseMaybeFloat(spinY) !== undefined &&
            parseMaybeFloat(spinZ) !== undefined
              ? {
                  x: parseMaybeFloat(spinX)!,
                  y: parseMaybeFloat(spinY)!,
                  z: parseMaybeFloat(spinZ)!,
                }
              : undefined,
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
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "380px 1fr",
        height: "100vh",
      }}
    >
      <div
        style={{
          borderRight: "1px solid #ddd",
          padding: 16,
          overflow: "auto",
        }}
      >
        <h1 style={{ margin: "0 0 12px 0", fontSize: 18 }}>BLCS Simulator</h1>
        {loadingCells ? (
          <div>Loading cells...</div>
        ) : cellsError ? (
          <div style={{ color: "crimson" }}>Cells error: {cellsError}</div>
        ) : null}

        <ControlsPanel
          fromSide={fromSide}
          setFromSide={(v) => {
            setFromSide(v);
            // Keep selection valid on side switch.
            setFromCell(0);
          }}
          fromCell={fromCell}
          setFromCell={setFromCell}
          fromCells={fromCells}
          targetMode={targetMode}
          setTargetMode={setTargetMode}
          toCell={toCell}
          setToCell={setToCell}
          toCells={toCells}
          posX={posX}
          setPosX={setPosX}
          posY={posY}
          setPosY={setPosY}
          posZ={posZ}
          setPosZ={setPosZ}
          velX={velX}
          setVelX={setVelX}
          velY={velY}
          setVelY={setVelY}
          velZ={velZ}
          setVelZ={setVelZ}
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
          running={running}
          onRun={onRun}
        />

        {simError ? <div style={{ color: "crimson" }}>{simError}</div> : null}
      </div>

      <div style={{ padding: 16, overflow: "auto" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 420px", gap: 16 }}>
          <div>
            <h2 style={{ margin: "0 0 8px 0", fontSize: 14 }}>Court (2D)</h2>
            <CourtPicker
              cells={cells}
              fromSide={fromSide}
              fromCell={fromCell}
              toCell={targetMode === "cell" ? toCell : null}
              targetSide={targetSide}
              onPickFrom={(id) => setFromCell(id)}
              onPickTo={(id) => setToCell(id)}
              pickMode={targetMode === "cell" ? "both" : "from"}
              trajectory={simResult?.positions ?? null}
              bounce1Pos={simResult?.events.bounce1_pos ?? null}
            />
            <div style={{ height: 8 }} />
            <Trajectory2D positions={simResult?.positions ?? null} />
          </div>

          <div>
            <h2 style={{ margin: "0 0 8px 0", fontSize: 14 }}>3D</h2>
            <Trajectory3D positions={simResult?.positions ?? null} />
            <div style={{ height: 16 }} />
            <MetricsPanel result={simResult} />
          </div>
        </div>
      </div>
    </div>
  );
}

