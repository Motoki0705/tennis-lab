import { useSceneStore } from "@/state/sceneStore";
import "@/styles/timeline.css";

export const Timeline = () => {
  const { scene, currentFrame, setFrame } = useSceneStore((state) => ({
    scene: state.scene,
    currentFrame: state.currentFrame,
    setFrame: state.setFrame,
  }));

  if (!scene) {
    return null;
  }

  const maxFrame = Math.max(0, scene.scene.num_frames - 1);
  const progress = maxFrame === 0 ? 0 : (currentFrame / maxFrame) * 100;

  return (
    <div className="timeline">
      <div className="timeline-header">
        <span>Frame {currentFrame + 1}</span>
        <span>/ {scene.scene.num_frames}</span>
      </div>
      <input
        type="range"
        min={0}
        max={maxFrame}
        value={currentFrame}
        onChange={(evt) => setFrame(Number(evt.target.value))}
      />
      <div className="timeline-progress">
        <div style={{ width: `${progress}%` }} />
      </div>
    </div>
  );
};
