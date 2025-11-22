import { Suspense, useEffect } from "react";
import { useSceneStore } from "@/state/sceneStore";
import { SceneCanvas } from "@/components/viewer3d/SceneCanvas";
import { Timeline } from "@/components/controls/Timeline";
import { TrackVisibilityPanel } from "@/components/controls/TrackVisibilityPanel";
import { SceneMetadataPanel } from "@/components/controls/SceneMetadataPanel";
import "@/styles/app.css";

function App() {
  const { loadSampleScene } = useSceneStore();

  useEffect(() => {
    loadSampleScene();
  }, [loadSampleScene]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <SceneMetadataPanel />
        <TrackVisibilityPanel />
      </aside>
      <main className="main">
        <div className="viewer">
          <Suspense fallback={<div className="viewer-fallback">Loading...</div>}>
            <SceneCanvas />
          </Suspense>
        </div>
        <Timeline />
      </main>
    </div>
  );
}

export default App;
