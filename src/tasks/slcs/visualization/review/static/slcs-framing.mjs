// Fit the court and every accepted teacher root into the task's narrow view.
export function fitOrbit(points, { yaw, pitch, fov, aspect, padding = 0.74 }) {
  if (!points.length || !Number.isFinite(aspect) || aspect <= 0 || !(fov > 0 && fov < 180)) throw new Error("Invalid scene fit inputs");
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const point of points) for (let k = 0; k < 3; k++) {
    if (!Number.isFinite(point[k])) throw new Error("Nonfinite scene fit point");
    min[k] = Math.min(min[k], point[k]); max[k] = Math.max(max[k], point[k]);
  }
  const target = min.map((v, k) => (v + max[k]) / 2);
  const cp = Math.cos(pitch), sp = Math.sin(pitch), cy = Math.cos(yaw), sy = Math.sin(yaw);
  const direction = [cp * cy, cp * sy, sp], right = [-sy, cy, 0], up = [-sp * cy, -sp * sy, cp];
  const tanY = Math.tan(fov * Math.PI / 360) * padding, tanX = tanY * aspect;
  let distance = 10;
  for (const point of points) {
    const q = point.map((v, k) => v - target[k]);
    const dot = (axis) => q.reduce((total, v, k) => total + v * axis[k], 0);
    distance = Math.max(distance, dot(direction) + Math.max(Math.abs(dot(right)) / tanX, Math.abs(dot(up)) / tanY) + 0.5);
  }
  return { yaw, pitch, distance, target };
}

export function framingPoints(model) {
  const points = model.court.keypoints.map((p) => [...p]);
  for (const entity of model.entities) for (let f = 0; f < entity.frames; f++) {
    if (entity.presence && entity.presence[f] !== 1) continue;
    points.push(Array.from(entity.roots.subarray(f * 3, f * 3 + 3)));
  }
  return points;
}
