import { useMemo } from "react";
import * as THREE from "three";

import type { CellInfo, CourtGeometryResponse, Side } from "../lib/types";

function makeLineSegmentsGeometry(
  keypoints: number[][],
  segments: number[][]
): THREE.BufferGeometry {
  const verts: number[] = [];
  for (const [i, j] of segments) {
    const a = keypoints[i];
    const b = keypoints[j];
    verts.push(a[0], a[1], a[2]);
    verts.push(b[0], b[1], b[2]);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  return g;
}

function CellHighlight(props: { cell: CellInfo; color: string; opacity: number }) {
  const b = props.cell.bounds;
  const w = b.x_max - b.x_min;
  const h = b.y_max - b.y_min;
  const cx = props.cell.center.x;
  const cy = props.cell.center.y;
  return (
    <mesh position={[cx, cy, 0.001]}>
      <planeGeometry args={[w, h]} />
      <meshStandardMaterial color={props.color} transparent opacity={props.opacity} />
    </mesh>
  );
}

export function Court3D(props: {
  court: CourtGeometryResponse | null;
  cells: CellInfo[];
  fromSide: Side;
  fromCell: number;
  toCell: number | null;
  targetSide: Side;
}) {
  const lines = useMemo(() => {
    if (!props.court) return null;
    const geom = makeLineSegmentsGeometry(props.court.keypoints, props.court.segments);
    const mat = new THREE.LineBasicMaterial({ color: 0xffffff });
    const obj = new THREE.LineSegments(geom, mat);
    obj.frustumCulled = false;
    return obj;
  }, [props.court]);

  const fromCellInfo = props.cells.find(
    (c) => c.side === props.fromSide && c.cell_id === props.fromCell
  );
  const toCellInfo =
    props.toCell !== null
      ? props.cells.find((c) => c.side === props.targetSide && c.cell_id === props.toCell)
      : undefined;

  return (
    <group>
      {/* Court ground */}
      <mesh position={[0, 0, 0]}>
        <planeGeometry args={[40, 70]} />
        <meshStandardMaterial color="#0b5" />
      </mesh>

      {/* Court lines */}
      {lines ? <primitive object={lines} /> : null}

      {/* Net (visual only) */}
      <mesh position={[0, 0, 0.6]} rotation={[Math.PI / 2, 0, 0]}>
        <planeGeometry args={[12, 1.2]} />
        <meshStandardMaterial color="#222" transparent opacity={0.15} />
      </mesh>

      {/* Cell highlights */}
      {fromCellInfo ? <CellHighlight cell={fromCellInfo} color="#111" opacity={0.28} /> : null}
      {toCellInfo ? <CellHighlight cell={toCellInfo} color="#0bf" opacity={0.22} /> : null}
    </group>
  );
}
