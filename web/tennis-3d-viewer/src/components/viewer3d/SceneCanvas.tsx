import { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useSceneStore } from "@/state/sceneStore";
import type { Vec3 } from "@/types/scene";

const CourtPlane = () => (
  <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
    <planeGeometry args={[23.77, 10.97]} />
    <meshStandardMaterial color="#2d7c4c" />
  </mesh>
);

const CourtLines = () => (
  <mesh rotation={[-Math.PI / 2, 0, 0]}>
    <planeGeometry args={[23.77, 10.97]} />
    <meshBasicMaterial color="#ffffff" wireframe />
  </mesh>
);

const PlayerTrack = ({ joints, color }: { joints: Vec3[]; color: string }) => (
  <group>
    {joints.map((joint, idx) => (
      <mesh key={`joint-${idx}`} position={[joint[0], joint[1], joint[2]]} castShadow>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
      </mesh>
    ))}
  </group>
);

export const SceneCanvas = () => {
  const { scene, currentFrame, visibleTrackIds } = useSceneStore();

  const frameTracks = useMemo(() => {
    if (!scene) {
      return [];
    }
    return scene.tracks
      .filter((track) => visibleTrackIds[track.id])
      .map((track) => {
        const frame = track.frames.find((f) => f.frame_index === currentFrame);
        if (!frame || !frame.joints_3d) {
          return null;
        }
        return {
          id: track.id,
          color: track.color_hint ?? "#ffb74d",
          joints: frame.joints_3d,
        };
      })
      .filter((entry): entry is { id: string; color: string; joints: Vec3[] } => Boolean(entry));
  }, [scene, currentFrame, visibleTrackIds]);

  if (!scene) {
    return <div className="viewer-placeholder">Load or drop a scene JSON to begin.</div>;
  }

  return (
    <div className="viewer-canvas">
      <Canvas shadows camera={{ position: [0, -15, 10], fov: 45 }}>
        <color attach="background" args={[0.03, 0.05, 0.07]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, -10, 12]} intensity={0.9} castShadow />
        <fog attach="fog" args={["#0a0a0a", 20, 80]} />
        <Suspense fallback={null}>
          <CourtPlane />
          <CourtLines />
          {frameTracks.map((track) => (
            <PlayerTrack key={track.id} joints={track.joints} color={track.color} />
          ))}
        </Suspense>
        <OrbitControls maxPolarAngle={Math.PI / 2.05} target={[0, 0, 1]} />
      </Canvas>
    </div>
  );
};
