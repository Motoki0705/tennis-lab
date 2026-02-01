import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";

import { Court3D } from "./Court3D";
import { FpsControls } from "./FpsControls";
import type { CellInfo, CourtGeometryResponse, Side, Vec3 } from "../lib/types";

function EventMarker(props: { pos: Vec3; color: string }) {
  return (
    <mesh position={[props.pos.x, props.pos.y, props.pos.z + 0.02]}>
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshStandardMaterial color={props.color} />
    </mesh>
  );
}

export function Trajectory3D(props: {
  positions: number[][] | null;
  court: CourtGeometryResponse | null;
  cells: CellInfo[];
  fromSide: Side;
  fromCell: number;
  toCell: number | null;
  targetSide: Side;
  cameraMode: "orbit" | "fps";
  fpsMoveSpeed: number;
  bounce1Pos: Vec3 | null;
  bounce2Pos: Vec3 | null;
  netPos: Vec3 | null;
}) {
  const trajectoryPoints = useMemo(() => {
    if (!props.positions || props.positions.length <= 1) return null;
    return props.positions.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
  }, [props.positions]);

  return (
    <div style={{ height: "100vh", width: "100vw" }}>
      <Canvas
        camera={{ position: [0, -18, 6], fov: 55, near: 0.05, far: 500, up: [0, 0, 1] }}
        onCreated={({ camera }) => {
          // Ensure the initial view is sane even before controls take over.
          // Use Z-up to match `src/utils/geometry/court.py`.
          camera.up.set(0, 0, 1);
          camera.lookAt(0, 0, 0);
        }}
      >
        <color attach="background" args={["#050708"]} />
        <ambientLight intensity={0.7} />
        <directionalLight position={[8, -4, 10]} intensity={0.9} />

        <Court3D
          court={props.court}
          cells={props.cells}
          fromSide={props.fromSide}
          fromCell={props.fromCell}
          toCell={props.toCell}
          targetSide={props.targetSide}
        />

        {trajectoryPoints ? (
          <Line
            points={trajectoryPoints}
            color="#e11d48"
            lineWidth={2}
          />
        ) : null}

        {/* Event markers */}
        {props.netPos ? <EventMarker pos={props.netPos} color="#fde047" /> : null}
        {props.bounce1Pos ? <EventMarker pos={props.bounce1Pos} color="#111" /> : null}
        {props.bounce2Pos ? <EventMarker pos={props.bounce2Pos} color="#111" /> : null}

        {props.cameraMode === "orbit" ? (
          <OrbitControls makeDefault target={[0, 0, 0]} />
        ) : (
          <FpsControls moveSpeed={props.fpsMoveSpeed} />
        )}
      </Canvas>
    </div>
  );
}
