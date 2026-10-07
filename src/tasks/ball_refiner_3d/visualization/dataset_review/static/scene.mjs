import { PRESETS, Scene3D, frustumFromParams } from "/shared/scene3d.mjs";

export const COLORS = { gt: "#48d597", input: "#a5b3c8", prediction: "#ff9a75", integrated: "#4fb3ff",
  integrated_truth: "#ff6fb5", linear: "#c08457", shot: "#bb9cf5", bounce: "#eec365" };
export const LABELS = { gt: "GT", input: "拡張後入力", prediction: "推論（直接）", integrated: "積分（予測区間）",
  integrated_truth: "積分（GT区間）", linear: "線形補間" };
// Trajectory series in drawing order; a scene carries each as `${kind}_3d` or null.
export const SERIES = ["gt", "input", "linear", "prediction", "integrated", "integrated_truth"];
export const present = (scene) => SERIES.filter((kind) => scene[`${kind}_3d`]);

export function canvases(container, separate, kinds) {
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
    const sameRally = this.rally === scene.rally;
    let refit = !sameRally || this.separate !== separate;
    this.rally = scene.rally;
    const points = [...scene.gt_3d, ...scene.court.keypoints];
    this.target = [0, 1, 2].map((axis) => (Math.min(...points.map((p) => p[axis])) + Math.max(...points.map((p) => p[axis]))) / 2);
    this.points = points;
    // Side by side, one panel per shown series; overlay keeps one panel for all.
    const kinds = present(scene).filter((kind) => !separate || visibility[kind]);
    const layout = separate ? `separate:${kinds.join(",")}` : "overlay";
    if (this.layout !== layout || !this.panels.length) {
      const old = this.panels[0]?.view;
      const keep = old && sameRally && this.separate === separate
        ? { position: old.camera.position.clone(), quaternion: old.camera.quaternion.clone(), target: old.controls.target.clone() } : null;
      this.panels.forEach(({ view }) => { view.dispose(); view.renderer.forceContextLoss(); });
      this.panels = canvases(this.container, separate, kinds).map(({ canvas, kind }) => ({ view: new Scene3D(canvas), kind }));
      this.separate = separate; this.layout = layout; refit = !keep;
      if (keep) this.panels.forEach(({ view }) => {
        view.resize(); view.camera.position.copy(keep.position); view.camera.quaternion.copy(keep.quaternion);
        view.controls.target.copy(keep.target); view.controls.update();
      });
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
      const entities = kinds.filter((id) => !kind || kind === id).flatMap((id) => {
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
      const {yaw, pitch} = PRESETS.corner;
      const forward = [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)];
      const right = [-Math.sin(yaw), Math.cos(yaw), 0];
      const up = [-Math.sin(pitch) * Math.cos(yaw), -Math.sin(pitch) * Math.sin(yaw), Math.cos(pitch)];
      const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
      const distance = Math.max(15, ...this.points.map(point => {
        const delta = point.map((value, i) => value - this.target[i]);
        return dot(delta, forward) + 1.15 * Math.max(Math.abs(dot(delta, right)) / Math.tan(halfHorizontal), Math.abs(dot(delta, up)) / Math.tan(halfVertical));
      }));
      view.setOrbit({ ...PRESETS.corner, target: this.target, distance });
    });
  }
}
