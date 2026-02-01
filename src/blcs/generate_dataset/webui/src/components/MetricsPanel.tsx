import type { SimulateShotResponse } from "../lib/types";

function Row(props: { label: string; value: string }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: 8, marginBottom: 6 }}>
      <div style={{ fontSize: 12, color: "rgba(255,255,255,0.8)" }}>{props.label}</div>
      <div
        style={{
          fontSize: 12,
          color: "#fff",
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
        }}
      >
        {props.value}
      </div>
    </div>
  );
}

export function MetricsPanel(props: { result: SimulateShotResponse | null }) {
  const r = props.result;
  if (!r) {
    return (
      <div
        style={{
          background: "rgba(10,10,10,0.78)",
          border: "1px solid rgba(255,255,255,0.14)",
          borderRadius: 14,
          padding: 12,
          color: "rgba(255,255,255,0.7)",
          fontSize: 12,
          backdropFilter: "blur(6px)",
        }}
      >
        Run a simulation to see metrics.
      </div>
    );
  }

  return (
    <div
      style={{
        background: "rgba(10,10,10,0.78)",
        border: "1px solid rgba(255,255,255,0.14)",
        borderRadius: 14,
        padding: 12,
        backdropFilter: "blur(6px)",
      }}
    >
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: "#fff" }}>Result</div>
      <Row label="category" value={r.labels.category} />
      <Row label="to_cell (classified)" value={r.labels.to_cell === null ? "null" : String(r.labels.to_cell)} />
      <Row label="apex_height_m" value={r.metrics.apex_height_m.toFixed(3)} />
      <Row
        label="time_to_bounce1_s"
        value={r.metrics.time_to_bounce1_s === null ? "null" : r.metrics.time_to_bounce1_s.toFixed(3)}
      />
      <Row
        label="net_clearance_m"
        value={r.metrics.net_clearance_m === null ? "null" : r.metrics.net_clearance_m.toFixed(3)}
      />
      <div style={{ height: 8 }} />
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: "#fff" }}>
        Events (frames @ output_fps)
      </div>
      <Row label="t_net" value={String(r.events.t_net)} />
      <Row label="t_bounce1" value={String(r.events.t_bounce1)} />
      <Row label="t_bounce2" value={String(r.events.t_bounce2)} />
    </div>
  );
}
