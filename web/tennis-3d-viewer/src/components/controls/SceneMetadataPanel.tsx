import { useSceneStore } from "@/state/sceneStore";
import "@/styles/panel.css";

export const SceneMetadataPanel = () => {
  const scene = useSceneStore((state) => state.scene);

  if (!scene) {
    return (
      <section className="panel">
        <h3>Scene Metadata</h3>
        <p>シーン JSON を読み込むと情報が表示されます。</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <h3>Scene Metadata</h3>
      <dl>
        <div>
          <dt>ID</dt>
          <dd>{scene.metadata.scene_id}</dd>
        </div>
        <div>
          <dt>Source</dt>
          <dd>{scene.metadata.source}</dd>
        </div>
        <div>
          <dt>Experiment</dt>
          <dd>{scene.metadata.experiment_name ?? "-"}</dd>
        </div>
        <div>
          <dt>FPS</dt>
          <dd>{scene.scene.fps}</dd>
        </div>
        <div>
          <dt>Cameras</dt>
          <dd>{scene.scene.num_cameras}</dd>
        </div>
      </dl>
    </section>
  );
};
