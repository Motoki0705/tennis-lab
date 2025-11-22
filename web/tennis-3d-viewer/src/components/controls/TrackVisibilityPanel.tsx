import { useSceneStore } from "@/state/sceneStore";
import "@/styles/panel.css";

export const TrackVisibilityPanel = () => {
  const { scene, visibleTrackIds, toggleTrackVisibility } = useSceneStore(
    (state) => ({
      scene: state.scene,
      visibleTrackIds: state.visibleTrackIds,
      toggleTrackVisibility: state.toggleTrackVisibility,
    })
  );

  if (!scene) {
    return null;
  }

  return (
    <section className="panel">
      <h3>Tracks</h3>
      <ul className="panel-list">
        {scene.tracks.map((track) => (
          <li key={track.id}>
            <label>
              <input
                type="checkbox"
                checked={visibleTrackIds[track.id]}
                onChange={() => toggleTrackVisibility(track.id)}
              />
              <span
                className="track-color"
                style={{ background: track.color_hint ?? "#ffd54f" }}
              />
              {track.label ?? track.id}
            </label>
          </li>
        ))}
      </ul>
    </section>
  );
};
