import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import { Court3D } from "./Court3D";
import { FpsControls } from "./FpsControls";
import type { CameraPreset, CellInfo, CourtGeometryResponse, Side, Vec3 } from "../lib/types";

function EventMarker(props: { pos: Vec3; color: string }) {
  return (
    <mesh position={[props.pos.x, props.pos.y, props.pos.z + 0.02]}>
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshStandardMaterial color={props.color} />
    </mesh>
  );
}

function CameraSync(props: {
  cameraPose: Vec3 | null;
  cameraLookAtTarget: Vec3 | null;
  cameraPoseVersion: number;
  lockLookAtCenter: boolean;
  onCameraPoseChange: (pos: Vec3, dir: Vec3) => void;
}) {
  const reportClock = useRef(0);
  const dir = useRef(new THREE.Vector3());
  const prevPos = useRef(new THREE.Vector3(Number.NaN, Number.NaN, Number.NaN));
  const prevDir = useRef(new THREE.Vector3(Number.NaN, Number.NaN, Number.NaN));
  const { camera } = useThree();

  useEffect(() => {
    if (!props.cameraPose) return;
    camera.position.set(props.cameraPose.x, props.cameraPose.y, props.cameraPose.z);
    if (props.lockLookAtCenter) {
      camera.lookAt(0, 0, 0);
    } else if (props.cameraLookAtTarget) {
      camera.lookAt(
        props.cameraLookAtTarget.x,
        props.cameraLookAtTarget.y,
        props.cameraLookAtTarget.z
      );
    }
    camera.updateProjectionMatrix();
  }, [
    camera,
    props.cameraPose,
    props.cameraLookAtTarget,
    props.cameraPoseVersion,
    props.lockLookAtCenter,
  ]);

  useFrame((_, dt: number) => {
    reportClock.current += dt;
    if (reportClock.current < 0.1) return;
    reportClock.current = 0;

    camera.getWorldDirection(dir.current);
    const p = camera.position;
    const d = dir.current;

    const posChanged = p.distanceToSquared(prevPos.current) > 1e-6;
    const dirChanged = d.distanceToSquared(prevDir.current) > 1e-6;
    if (!posChanged && !dirChanged) return;

    prevPos.current.copy(p);
    prevDir.current.copy(d);
    props.onCameraPoseChange(
      { x: p.x, y: p.y, z: p.z },
      { x: d.x, y: d.y, z: d.z }
    );
  });

  return null;
}

function CameraMarker(props: { pos: Vec3; lookAt: Vec3; active: boolean }) {
  const quaternion = useMemo(() => {
    const position = new THREE.Vector3(props.pos.x, props.pos.y, props.pos.z);
    const target = new THREE.Vector3(props.lookAt.x, props.lookAt.y, props.lookAt.z);
    const forward = target.sub(position);
    if (forward.lengthSq() < 1e-6) {
      forward.set(1, 0, 0);
    } else {
      forward.normalize();
    }

    const worldUp = new THREE.Vector3(0, 0, 1);
    const right = new THREE.Vector3().crossVectors(worldUp, forward);
    if (right.lengthSq() < 1e-6) {
      right.set(0, 1, 0).cross(forward);
    }
    right.normalize();

    const up = new THREE.Vector3().crossVectors(forward, right).normalize();
    const basis = new THREE.Matrix4().makeBasis(forward, right, up);
    return new THREE.Quaternion().setFromRotationMatrix(basis);
  }, [props.lookAt.x, props.lookAt.y, props.lookAt.z, props.pos.x, props.pos.y, props.pos.z]);

  const bodyColor = props.active ? "#fde047" : "#f97316";
  const accentColor = props.active ? "#fff7c2" : "#fed7aa";

  return (
    <group
      position={[props.pos.x, props.pos.y, props.pos.z]}
      quaternion={quaternion}
      scale={props.active ? 1.16 : 1}
    >
      <mesh>
        <boxGeometry args={[0.5, 0.3, 0.26]} />
        <meshStandardMaterial color={bodyColor} roughness={0.42} metalness={0.08} />
      </mesh>

      <mesh position={[0.34, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.085, 0.11, 0.28, 24]} />
        <meshStandardMaterial color="#111827" roughness={0.35} metalness={0.22} />
      </mesh>

      <mesh position={[0.47, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.05, 0.07, 0.08, 24]} />
        <meshStandardMaterial color={accentColor} transparent opacity={0.92} roughness={0.12} />
      </mesh>

      <mesh position={[-0.04, 0, 0.2]}>
        <boxGeometry args={[0.18, 0.18, 0.09]} />
        <meshStandardMaterial color={accentColor} roughness={0.48} metalness={0.06} />
      </mesh>

      <mesh position={[-0.08, 0, -0.2]}>
        <boxGeometry args={[0.16, 0.18, 0.1]} />
        <meshStandardMaterial color="#0f172a" roughness={0.7} metalness={0.04} />
      </mesh>
    </group>
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
  cameraPose: Vec3 | null;
  cameraLookAtTarget: Vec3 | null;
  cameraPoseVersion: number;
  lockLookAtCenter: boolean;
  activePresetId: string | null;
  onCameraPoseChange: (pos: Vec3, dir: Vec3) => void;
  cameraMarkers: CameraPreset[];
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

        {props.cameraMarkers.map((marker) => {
          const tooCloseToViewer =
            props.cameraPose !== null &&
            (props.cameraPose.x - marker.pos.x) ** 2 +
              (props.cameraPose.y - marker.pos.y) ** 2 +
              (props.cameraPose.z - marker.pos.z) ** 2 <
              0.9 ** 2;
          if (tooCloseToViewer) {
            return null;
          }

          const isActive = props.activePresetId === marker.id;
          return (
            <CameraMarker
              key={marker.id}
              pos={marker.pos}
              lookAt={marker.lookAt}
              active={isActive}
            />
          );
        })}

        <CameraSync
          cameraPose={props.cameraPose}
          cameraLookAtTarget={props.cameraLookAtTarget}
          cameraPoseVersion={props.cameraPoseVersion}
          lockLookAtCenter={props.lockLookAtCenter}
          onCameraPoseChange={props.onCameraPoseChange}
        />

        {props.cameraMode === "orbit" ? (
          <OrbitControls makeDefault target={[0, 0, 0]} />
        ) : (
          <FpsControls moveSpeed={props.fpsMoveSpeed} />
        )}
      </Canvas>
    </div>
  );
}
