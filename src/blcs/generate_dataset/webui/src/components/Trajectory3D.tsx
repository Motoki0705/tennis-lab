import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";

export function Trajectory3D(props: { positions: number[][] | null }) {
  return (
    <div style={{ height: 320, border: "1px solid #ddd", borderRadius: 12, overflow: "hidden" }}>
      <Canvas camera={{ position: [10, -10, 8], fov: 45 }}>
        <ambientLight intensity={0.7} />
        <directionalLight position={[8, -4, 10]} intensity={0.9} />

        {/* Ground plane */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
          <planeGeometry args={[40, 60]} />
          <meshStandardMaterial color="#f5f5f5" />
        </mesh>

        {/* Axes helper for orientation */}
        <axesHelper args={[3]} />

        {props.positions && props.positions.length > 1 ? (
          <Line
            points={props.positions.map((p) => new THREE.Vector3(p[0], p[1], p[2]))}
            color="#e11d48"
            lineWidth={2}
          />
        ) : null}

        <OrbitControls makeDefault />
      </Canvas>
    </div>
  );
}
