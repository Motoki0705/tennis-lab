export function Trajectory2D(props: { positions: number[][] | null }) {
  if (!props.positions || props.positions.length === 0) {
    return (
      <div
        style={{
          border: "1px dashed #ddd",
          borderRadius: 12,
          padding: 12,
          color: "#666",
          fontSize: 12,
        }}
      >
        Run a simulation to see the trajectory preview.
      </div>
    );
  }

  const pts = props.positions;
  const z = pts.map((p) => p[2]);
  const zMin = Math.min(...z);
  const zMax = Math.max(...z);

  return (
    <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
      <div style={{ fontSize: 12, color: "#444", marginBottom: 6 }}>Height summary</div>
      <div style={{ fontSize: 12, color: "#111" }}>
        z_min={zMin.toFixed(2)}m, z_max={zMax.toFixed(2)}m, frames={pts.length}
      </div>
    </div>
  );
}

