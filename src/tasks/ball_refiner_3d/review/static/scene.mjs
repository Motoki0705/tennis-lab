import { PRESETS, Scene3D, frustumFromParams } from "/shared/scene3d.mjs";

export const COLORS = { gt: "#48d597", input: "#a5b3c8", prediction: "#ff9a75", shot: "#bb9cf5", bounce: "#eec365" };
export const LABELS = { gt: "GT", input: "拡張後入力", prediction: "推論" };

export function canvases(container, separate, kinds = ["gt", "input", "prediction"]) {
  container.replaceChildren();
  return (separate ? kinds : [null]).map((kind) => {
    const wrapper = document.createElement("div");
    wrapper.className = "canvas-wrap";
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-label", `${container.id} ${kind ? LABELS[kind] : "比較"}`);
    wrapper.append(canvas);
    if (kind) {
      const label = document.createElement("span");
      label.className = "series-title";
      label.style.color = COLORS[kind];
      label.textContent = LABELS[kind];
      wrapper.append(label);
    }
    container.append(wrapper);
    return { canvas, kind };
  });
}

export class ScenePanels {
  constructor(container) { this.container = container; this.panels = []; this.syncing = false; }

  setup(scene, separate, visibility, showCameras) {
    const refit = this.rally !== scene.rally || this.separate !== separate;
    this.rally = scene.rally;
    const points = [...scene.gt_3d, ...scene.court.keypoints];
    this.target = [0, 1, 2].map((axis) => (Math.min(...points.map((p) => p[axis])) + Math.max(...points.map((p) => p[axis]))) / 2);
    this.radius = Math.max(...points.map((point) => Math.hypot(...point.map((value, axis) => value - this.target[axis]))));
    if (this.separate !== separate || !this.panels.length) {
      this.panels.forEach(({ view }) => { view.dispose(); view.renderer.forceContextLoss(); });
      this.panels = canvases(this.container, separate).map(({ canvas, kind }) => ({ view: new Scene3D(canvas), kind }));
      this.separate = separate;
      this.panels.forEach(({ view }) => view.controls.addEventListener("change", () => {
        if (this.syncing) return;
        this.syncing = true;
        this.panels.forEach(({ view: other }) => {
          if (other === view) return;
          other.camera.position.copy(view.camera.position);
          other.camera.quaternion.copy(view.camera.quaternion);
          other.controls.target.copy(view.controls.target);
          other.controls.update();
        });
        this.syncing = false;
      }));
    }
    this.panels.forEach(({ view, kind }) => {
      const entities = ["gt", "input", "prediction"].filter((id) => !kind || kind === id).flatMap((id) => {
        const points = scene[`${id}_3d`];
        if (!points) return [];
        const positions = Float32Array.from(points.flat());
        return [{ id, color: COLORS[id], kind: "ball", frames: scene.frames, joints: 1, positions, roots: positions,
          presence: Uint8Array.from(scene.missing_3d, (missing) => id === "input" ? !missing : 1), radius: 0.115 }];
      });
      view.setModel({ court: scene.court, frames: scene.frames, entities,
        cameras: scene.cameras.map((camera) => ({ id: camera.id, label: camera.label,
          center: camera.params.C, rotation: camera.params.R, frustum: frustumFromParams(camera.params, 2) })) });
      view.setCamerasVisible(showCameras);
      view.setTrailMode("full");
      Object.entries(visibility).forEach(([id, value]) => view.setEntityVisible(id, value));
    });
    if (refit) this.reset();
  }

  setFrame(frame) { this.panels.forEach(({ view }) => view.setFrame(frame)); }
  render() { this.panels.forEach(({ view }) => view.render()); }
  visibility(values) { this.panels.forEach(({ view }) => Object.entries(values).forEach(([id, visible]) => view.setEntityVisible(id, visible))); }
  cameras(visible) { this.panels.forEach(({ view }) => view.setCamerasVisible(visible)); }
  reset() {
    this.panels.forEach(({ view }) => {
      view.resize();
      const halfVertical = view.camera.fov * Math.PI / 360;
      const halfHorizontal = Math.atan(Math.tan(halfVertical) * view.camera.aspect);
      view.setOrbit({ ...PRESETS.corner, target: this.target, distance: Math.max(30, this.radius / Math.sin(Math.min(halfVertical, halfHorizontal)) * 1.1) });
    });
  }
}
