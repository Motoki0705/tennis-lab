import type { CellInfo, Side, TargetMode } from "../lib/types";

function Row(props: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 8, marginBottom: 8 }}>
      <div style={{ fontSize: 12, color: "#444" }}>{props.label}</div>
      <div>{props.children}</div>
    </div>
  );
}

function SmallInput(props: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <input
      value={props.value}
      placeholder={props.placeholder}
      onChange={(e) => props.onChange(e.target.value)}
      style={{
        width: "100%",
        boxSizing: "border-box",
        padding: "6px 8px",
        border: "1px solid #ccc",
        borderRadius: 6,
      }}
    />
  );
}

export function ControlsPanel(props: {
  fromSide: Side;
  setFromSide: (s: Side) => void;
  fromCell: number;
  setFromCell: (id: number) => void;
  fromCells: CellInfo[];

  targetMode: TargetMode;
  setTargetMode: (m: TargetMode) => void;
  toCell: number;
  setToCell: (id: number) => void;
  toCells: CellInfo[];

  posX: string;
  setPosX: (v: string) => void;
  posY: string;
  setPosY: (v: string) => void;
  posZ: string;
  setPosZ: (v: string) => void;

  velX: string;
  setVelX: (v: string) => void;
  velY: string;
  setVelY: (v: string) => void;
  velZ: string;
  setVelZ: (v: string) => void;

  spinX: string;
  setSpinX: (v: string) => void;
  spinY: string;
  setSpinY: (v: string) => void;
  spinZ: string;
  setSpinZ: (v: string) => void;

  useDrag: boolean;
  setUseDrag: (v: boolean) => void;
  useMagnus: boolean;
  setUseMagnus: (v: boolean) => void;

  running: boolean;
  onRun: () => void;
}) {
  return (
    <div>
      <div style={{ padding: "10px 10px", background: "#fafafa", border: "1px solid #eee", borderRadius: 10 }}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>Origin / Target</div>

        <Row label="from_side">
          <select
            value={props.fromSide}
            onChange={(e) => props.setFromSide(e.target.value as Side)}
            style={{ width: "100%", padding: "6px 8px", borderRadius: 6 }}
          >
            <option value="near">near</option>
            <option value="far">far</option>
          </select>
        </Row>

        <Row label="from_cell">
          <select
            value={props.fromCell}
            onChange={(e) => props.setFromCell(Number(e.target.value))}
            style={{ width: "100%", padding: "6px 8px", borderRadius: 6 }}
          >
            {props.fromCells.map((c) => (
              <option key={`${c.side}-${c.cell_id}`} value={c.cell_id}>
                {c.cell_id}
              </option>
            ))}
          </select>
        </Row>

        <Row label="target_mode">
          <select
            value={props.targetMode}
            onChange={(e) => props.setTargetMode(e.target.value as TargetMode)}
            style={{ width: "100%", padding: "6px 8px", borderRadius: 6 }}
          >
            <option value="none">none</option>
            <option value="cell">cell</option>
            <option value="point" disabled>
              point (TODO)
            </option>
          </select>
        </Row>

        {props.targetMode === "cell" ? (
          <Row label="to_cell">
            <select
              value={props.toCell}
              onChange={(e) => props.setToCell(Number(e.target.value))}
              style={{ width: "100%", padding: "6px 8px", borderRadius: 6 }}
            >
              {props.toCells.map((c) => (
                <option key={`${c.side}-${c.cell_id}`} value={c.cell_id}>
                  {c.cell_id}
                </option>
              ))}
            </select>
          </Row>
        ) : null}
      </div>

      <div style={{ height: 12 }} />

      <div style={{ padding: "10px 10px", background: "#fafafa", border: "1px solid #eee", borderRadius: 10 }}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>Initial State Overrides</div>

        <Row label="position (m)">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
            <SmallInput value={props.posX} onChange={props.setPosX} placeholder="x" />
            <SmallInput value={props.posY} onChange={props.setPosY} placeholder="y" />
            <SmallInput value={props.posZ} onChange={props.setPosZ} placeholder="z" />
          </div>
        </Row>

        <Row label="velocity (m/s)">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
            <SmallInput value={props.velX} onChange={props.setVelX} placeholder="vx" />
            <SmallInput value={props.velY} onChange={props.setVelY} placeholder="vy" />
            <SmallInput value={props.velZ} onChange={props.setVelZ} placeholder="vz" />
          </div>
        </Row>

        <Row label="spin (rad/s)">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
            <SmallInput value={props.spinX} onChange={props.setSpinX} placeholder="wx" />
            <SmallInput value={props.spinY} onChange={props.setSpinY} placeholder="wy" />
            <SmallInput value={props.spinZ} onChange={props.setSpinZ} placeholder="wz" />
          </div>
        </Row>
      </div>

      <div style={{ height: 12 }} />

      <div style={{ padding: "10px 10px", background: "#fafafa", border: "1px solid #eee", borderRadius: 10 }}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>Physics</div>

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
      </div>

      <div style={{ height: 12 }} />

      <button
        onClick={props.onRun}
        disabled={props.running}
        style={{
          width: "100%",
          padding: "10px 12px",
          borderRadius: 10,
          border: "1px solid #111",
          background: props.running ? "#999" : "#111",
          color: "white",
          fontWeight: 600,
          cursor: props.running ? "not-allowed" : "pointer",
        }}
      >
        {props.running ? "Running..." : "Run Simulation"}
      </button>
    </div>
  );
}

