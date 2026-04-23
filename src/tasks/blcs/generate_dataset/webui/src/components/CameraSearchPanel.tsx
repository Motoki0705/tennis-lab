import type { CameraPreset, Vec3 } from "../lib/types";

function fmt(v: number) {
  return v.toFixed(3);
}

export function CameraSearchPanel(props: {
  presets: CameraPreset[];
  activePresetId: string | null;
  zMin: number;
  setZMin: (v: number) => void;
  zMax: number;
  setZMax: (v: number) => void;
  lockLookAtCenter: boolean;
  setLockLookAtCenter: (v: boolean) => void;
  onApplyPreset: (presetId: string) => void;
  cameraPos: Vec3 | null;
  cameraDir: Vec3 | null;
}) {
  return (
    <div
      style={{
        position: "fixed",
        right: 16,
        bottom: 16,
        zIndex: 9,
        width: 360,
        background: "rgba(10,10,10,0.78)",
        border: "1px solid rgba(255,255,255,0.14)",
        borderRadius: 14,
        padding: 14,
        backdropFilter: "blur(6px)",
        color: "#fff",
        boxShadow: "0 20px 40px rgba(0,0,0,0.35)",
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: 10 }}>Camera Search</div>

      <div style={{ fontSize: 12, opacity: 0.85, marginBottom: 8 }}>
        Presets: 4 corners at z_min, 4 edge-midpoints at z_max.
      </div>

      <Row label="z_min">
        <Slider
          min={1.0}
          max={15.0}
          step={0.1}
          value={props.zMin}
          onChange={props.setZMin}
          format={(v) => v.toFixed(1)}
        />
      </Row>
      <Row label="z_max">
        <Slider
          min={1.0}
          max={20.0}
          step={0.1}
          value={props.zMax}
          onChange={props.setZMax}
          format={(v) => v.toFixed(1)}
        />
      </Row>

      <Row label="look_at">
        <label style={{ fontSize: 12 }}>
          <input
            type="checkbox"
            checked={props.lockLookAtCenter}
            onChange={(e) => props.setLockLookAtCenter(e.target.checked)}
            style={{ marginRight: 8 }}
          />
          always (0, 0, 0)
        </label>
      </Row>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 8, marginTop: 8 }}>
        {props.presets.map((preset) => {
          const active = props.activePresetId === preset.id;
          return (
            <button
              key={preset.id}
              onClick={() => props.onApplyPreset(preset.id)}
              style={{
                border: "1px solid rgba(255,255,255,0.25)",
                background: active ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.08)",
                color: "#fff",
                borderRadius: 10,
                padding: "8px 10px",
                fontSize: 12,
                cursor: "pointer",
              }}
            >
              {preset.label}
            </button>
          );
        })}
      </div>

      <div
        style={{
          marginTop: 12,
          padding: 10,
          borderRadius: 10,
          border: "1px solid rgba(255,255,255,0.16)",
          background: "rgba(255,255,255,0.04)",
          fontSize: 12,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          lineHeight: 1.6,
        }}
      >
        <div>
          cam_pos:{" "}
          {props.cameraPos
            ? `(${fmt(props.cameraPos.x)}, ${fmt(props.cameraPos.y)}, ${fmt(props.cameraPos.z)})`
            : "-"}
        </div>
        <div>
          cam_dir:{" "}
          {props.cameraDir
            ? `(${fmt(props.cameraDir.x)}, ${fmt(props.cameraDir.y)}, ${fmt(props.cameraDir.z)})`
            : "-"}
        </div>
      </div>
    </div>
  );
}

function Row(props: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "80px 1fr", gap: 10, marginBottom: 8 }}>
      <div style={{ fontSize: 12, opacity: 0.85 }}>{props.label}</div>
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
  const format = props.format ?? ((v: number) => String(v));
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 56px", gap: 8, alignItems: "center" }}>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
      <div style={{ fontSize: 12, textAlign: "right", opacity: 0.95 }}>{format(props.value)}</div>
    </div>
  );
}
