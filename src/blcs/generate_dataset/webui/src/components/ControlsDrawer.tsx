import type { Side, TargetMode } from "../lib/types";

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function Row(props: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "130px 1fr", gap: 10, marginBottom: 10 }}>
      <div style={{ fontSize: 12, color: "#e8e8e8", opacity: 0.9 }}>{props.label}</div>
      <div>{props.children}</div>
    </div>
  );
}

function Slider(props: {
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  const fmt = props.format ?? ((v: number) => String(v));
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 72px", gap: 10, alignItems: "center" }}>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
        style={{ width: "100%" }}
      />
      <div
        style={{
          fontSize: 12,
          color: "#fff",
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          textAlign: "right",
        }}
      >
        {fmt(props.value)}
      </div>
    </div>
  );
}

export function ControlsDrawer(props: {
  open: boolean;
  setOpen: (v: boolean) => void;

  fromSide: Side;
  setFromSide: (s: Side) => void;
  fromCell: number;
  setFromCell: (id: number) => void;

  targetMode: TargetMode;
  setTargetMode: (m: TargetMode) => void;
  toCell: number;
  setToCell: (id: number) => void;

  offsetX: number;
  setOffsetX: (v: number) => void;
  offsetY: number;
  setOffsetY: (v: number) => void;
  z0: number;
  setZ0: (v: number) => void;

  speed: number;
  setSpeed: (v: number) => void;
  azimuthDeg: number;
  setAzimuthDeg: (v: number) => void;
  elevationDeg: number;
  setElevationDeg: (v: number) => void;

  spinX: number;
  setSpinX: (v: number) => void;
  spinY: number;
  setSpinY: (v: number) => void;
  spinZ: number;
  setSpinZ: (v: number) => void;

  useDrag: boolean;
  setUseDrag: (v: boolean) => void;
  useMagnus: boolean;
  setUseMagnus: (v: boolean) => void;

  cameraMode: "orbit" | "fps";
  setCameraMode: (v: "orbit" | "fps") => void;
  fpsMoveSpeed: number;
  setFpsMoveSpeed: (v: number) => void;

  running: boolean;
  onRun: () => void;
}) {
  const drawerW = 360;

  return (
    <div
      style={{
        position: "fixed",
        top: 16,
        left: 16,
        width: drawerW,
        transform: props.open ? "translateX(0)" : `translateX(calc(-${drawerW}px - 16px))`,
        transition: "transform 120ms ease",
        background: "rgba(10,10,10,0.78)",
        border: "1px solid rgba(255,255,255,0.14)",
        borderRadius: 14,
        padding: 14,
        backdropFilter: "blur(6px)",
        color: "#fff",
        boxShadow: "0 20px 40px rgba(0,0,0,0.35)",
        zIndex: 10,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div style={{ fontWeight: 700 }}>Controls</div>
        <button
          onClick={() => props.setOpen(false)}
          style={{
            border: "1px solid rgba(255,255,255,0.25)",
            background: "transparent",
            color: "#fff",
            borderRadius: 10,
            padding: "6px 10px",
            cursor: "pointer",
          }}
        >
          Close
        </button>
      </div>

      <div style={{ maxHeight: "78vh", overflow: "auto", paddingRight: 6 }}>
        <div style={{ fontSize: 12, opacity: 0.9, marginBottom: 12 }}>
          FPS mode: click canvas to lock pointer, ESC to unlock.
        </div>

        <Row label="camera_mode">
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={() => props.setCameraMode("orbit")}
              style={pill(props.cameraMode === "orbit")}
            >
              orbit
            </button>
            <button
              onClick={() => props.setCameraMode("fps")}
              style={pill(props.cameraMode === "fps")}
            >
              fps
            </button>
          </div>
        </Row>

        {props.cameraMode === "fps" ? (
          <Row label="fps_move_speed">
            <Slider
              min={1}
              max={20}
              step={0.5}
              value={props.fpsMoveSpeed}
              onChange={props.setFpsMoveSpeed}
              format={(v) => v.toFixed(1)}
            />
          </Row>
        ) : null}

        <div style={sectionTitle()}>Origin / Target</div>

        <Row label="from_side">
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={() => props.setFromSide("near")} style={pill(props.fromSide === "near")}>
              near
            </button>
            <button onClick={() => props.setFromSide("far")} style={pill(props.fromSide === "far")}>
              far
            </button>
          </div>
        </Row>

        <Row label="from_cell">
          <Slider
            min={0}
            max={19}
            step={1}
            value={props.fromCell}
            onChange={(v) => props.setFromCell(Math.round(v))}
            format={(v) => String(Math.round(v))}
          />
        </Row>

        <Row label="target_mode">
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={() => props.setTargetMode("none")} style={pill(props.targetMode === "none")}>
              none
            </button>
            <button onClick={() => props.setTargetMode("cell")} style={pill(props.targetMode === "cell")}>
              cell
            </button>
            <button onClick={() => props.setTargetMode("point")} style={{ ...pill(false), opacity: 0.4 }} disabled>
              point (TODO)
            </button>
          </div>
        </Row>

        {props.targetMode === "cell" ? (
          <Row label="to_cell">
            <Slider
              min={0}
              max={19}
              step={1}
              value={props.toCell}
              onChange={(v) => props.setToCell(Math.round(v))}
              format={(v) => String(Math.round(v))}
            />
          </Row>
        ) : null}

        <div style={sectionTitle()}>Start Position (in cell)</div>
        <Row label="offset_x">
          <Slider min={0} max={1} step={0.01} value={props.offsetX} onChange={(v) => props.setOffsetX(clamp01(v))} format={(v) => v.toFixed(2)} />
        </Row>
        <Row label="offset_y">
          <Slider min={0} max={1} step={0.01} value={props.offsetY} onChange={(v) => props.setOffsetY(clamp01(v))} format={(v) => v.toFixed(2)} />
        </Row>
        <Row label="z0 (m)">
          <Slider min={0.3} max={3.0} step={0.01} value={props.z0} onChange={props.setZ0} format={(v) => v.toFixed(2)} />
        </Row>

        <div style={sectionTitle()}>Launch</div>
        <Row label="speed (m/s)">
          <Slider min={5} max={45} step={0.1} value={props.speed} onChange={props.setSpeed} format={(v) => v.toFixed(1)} />
        </Row>
        <Row label="azimuth (deg)">
          <Slider min={-60} max={60} step={0.1} value={props.azimuthDeg} onChange={props.setAzimuthDeg} format={(v) => v.toFixed(1)} />
        </Row>
        <Row label="elevation (deg)">
          <Slider min={0} max={75} step={0.1} value={props.elevationDeg} onChange={props.setElevationDeg} format={(v) => v.toFixed(1)} />
        </Row>

        <div style={sectionTitle()}>Spin (rad/s)</div>
        <Row label="wx">
          <Slider min={-120} max={120} step={1} value={props.spinX} onChange={props.setSpinX} format={(v) => v.toFixed(0)} />
        </Row>
        <Row label="wy">
          <Slider min={-120} max={120} step={1} value={props.spinY} onChange={props.setSpinY} format={(v) => v.toFixed(0)} />
        </Row>
        <Row label="wz">
          <Slider min={-120} max={120} step={1} value={props.spinZ} onChange={props.setSpinZ} format={(v) => v.toFixed(0)} />
        </Row>

        <div style={sectionTitle()}>Physics</div>
        <Row label="use_drag">
          <label style={{ fontSize: 12 }}>
            <input
              type="checkbox"
              checked={props.useDrag}
              onChange={(e) => props.setUseDrag(e.target.checked)}
              style={{ marginRight: 8 }}
            />
            enabled
          </label>
        </Row>
        <Row label="use_magnus">
          <label style={{ fontSize: 12 }}>
            <input
              type="checkbox"
              checked={props.useMagnus}
              onChange={(e) => props.setUseMagnus(e.target.checked)}
              style={{ marginRight: 8 }}
            />
            enabled
          </label>
        </Row>

        <button
          onClick={props.onRun}
          disabled={props.running}
          style={{
            width: "100%",
            padding: "10px 12px",
            borderRadius: 12,
            border: "1px solid rgba(255,255,255,0.25)",
            background: props.running ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.12)",
            color: "white",
            fontWeight: 700,
            cursor: props.running ? "not-allowed" : "pointer",
          }}
        >
          {props.running ? "Running..." : "Run Simulation"}
        </button>
      </div>
    </div>
  );
}

function pill(active: boolean): React.CSSProperties {
  return {
    border: "1px solid rgba(255,255,255,0.25)",
    background: active ? "rgba(255,255,255,0.20)" : "rgba(255,255,255,0.08)",
    color: "#fff",
    borderRadius: 999,
    padding: "6px 10px",
    cursor: "pointer",
    fontSize: 12,
    fontWeight: 650,
  };
}

function sectionTitle(): React.CSSProperties {
  return { fontSize: 12, fontWeight: 800, margin: "14px 0 10px 0", opacity: 0.9 };
}

