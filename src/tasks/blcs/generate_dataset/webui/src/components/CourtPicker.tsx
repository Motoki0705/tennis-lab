import type { CellInfo, Side, Vec3 } from "../lib/types";

type PickMode = "from" | "both";

function getExtents(cells: CellInfo[]) {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const c of cells) {
    xMin = Math.min(xMin, c.bounds.x_min);
    xMax = Math.max(xMax, c.bounds.x_max);
    yMin = Math.min(yMin, c.bounds.y_min);
    yMax = Math.max(yMax, c.bounds.y_max);
  }
  return { xMin, xMax, yMin, yMax };
}

export function CourtPicker(props: {
  cells: CellInfo[];
  fromSide: Side;
  targetSide: Side;
  fromCell: number;
  toCell: number | null;
  pickMode: PickMode;
  onPickFrom: (cellId: number) => void;
  onPickTo: (cellId: number) => void;
  trajectory: number[][] | null;
  bounce1Pos: Vec3 | null;
}) {
  const W = 700;
  const H = 420;

  const ext = getExtents(props.cells);
  const pad = 8;

  function wx(x: number) {
    return (
      pad +
      ((x - ext.xMin) / (ext.xMax - ext.xMin + 1e-9)) * (W - pad * 2)
    );
  }
  function wy(y: number) {
    // SVG y is down; court y increases upward.
    return (
      pad +
      (1 - (y - ext.yMin) / (ext.yMax - ext.yMin + 1e-9)) * (H - pad * 2)
    );
  }

  const trajPath =
    props.trajectory && props.trajectory.length > 1
      ? "M " +
        props.trajectory
          .map((p) => `${wx(p[0]).toFixed(2)} ${wy(p[1]).toFixed(2)}`)
          .join(" L ")
      : null;

  return (
    <svg
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      style={{ border: "1px solid #ddd", borderRadius: 12, background: "#fff" }}
    >
      {props.cells.map((c) => {
        const isFrom = c.side === props.fromSide && c.cell_id === props.fromCell;
        const isTo =
          props.toCell !== null && c.side === props.targetSide && c.cell_id === props.toCell;
        const selectable =
          (props.pickMode === "from" && c.side === props.fromSide) ||
          (props.pickMode === "both" &&
            ((c.side === props.fromSide) || (c.side === props.targetSide)));

        const fill = isFrom ? "#111" : isTo ? "#0b6" : c.side === "near" ? "#f7f7f7" : "#f3fbff";
        const stroke = c.side === "near" ? "#bbb" : "#b6d7ff";
        const x = wx(c.bounds.x_min);
        const y = wy(c.bounds.y_max);
        const w = wx(c.bounds.x_max) - wx(c.bounds.x_min);
        const h = wy(c.bounds.y_min) - wy(c.bounds.y_max);

        return (
          <g key={`${c.side}-${c.cell_id}`}>
            <rect
              x={x}
              y={y}
              width={w}
              height={h}
              fill={fill}
              stroke={stroke}
              strokeWidth={1}
              opacity={selectable ? 1 : 0.35}
              style={{ cursor: selectable ? "pointer" : "default" }}
              onClick={() => {
                if (!selectable) return;
                if (c.side === props.fromSide) props.onPickFrom(c.cell_id);
                else props.onPickTo(c.cell_id);
              }}
            />
            <text
              x={x + 6}
              y={y + 14}
              fontSize={12}
              fill={isFrom || isTo ? "#fff" : "#333"}
              opacity={selectable ? 1 : 0.35}
            >
              {c.cell_id}
            </text>
          </g>
        );
      })}

      {/* Net line at y=0 */}
      <line
        x1={wx(ext.xMin)}
        y1={wy(0)}
        x2={wx(ext.xMax)}
        y2={wy(0)}
        stroke="#000"
        strokeWidth={2}
        opacity={0.5}
      />

      {/* Trajectory overlay */}
      {trajPath ? (
        <path d={trajPath} fill="none" stroke="#e11d48" strokeWidth={2} opacity={0.9} />
      ) : null}

      {/* First bounce marker */}
      {props.bounce1Pos ? (
        <circle
          cx={wx(props.bounce1Pos.x)}
          cy={wy(props.bounce1Pos.y)}
          r={5}
          fill="#111"
          stroke="#fff"
          strokeWidth={2}
        />
      ) : null}
    </svg>
  );
}

