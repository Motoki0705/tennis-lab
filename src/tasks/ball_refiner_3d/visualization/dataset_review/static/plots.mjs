import { COLORS } from "./scene.mjs";

export function surface(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const w = Math.max(1, rect.width), h = Math.max(1, rect.height);
  if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
    canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.fillStyle = "#0d1725"; ctx.fillRect(0, 0, w, h);
  ctx.font = "10px system-ui"; ctx.lineWidth = 1;
  return { ctx, w, h };
}

function line(ctx, points, project, valid = () => true) {
  ctx.beginPath(); let connected = false;
  points.forEach((value, i) => {
    if (value === null || !valid(i)) { connected = false; return; }
    const [x, y] = project(value, i);
    if (!Number.isFinite(x) || !Number.isFinite(y)) { connected = false; return; }
    if (connected) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    connected = true;
  });
  ctx.stroke();
}

export function drawTimeline(canvas, scene, frame) {
  const { ctx, w, h } = surface(canvas), left = 47, right = w - 12;
  const x = (i) => left + i / Math.max(1, scene.frames - 1) * (right - left);
  const dx = Math.max(1, (right - left) / scene.frames);
  ctx.fillStyle = "#899fb6"; ctx.fillText("Event", 8, 15); ctx.fillText("3D", 8, 40);
  for (let i = 0; i < scene.frames; i++) {
    if (scene.events[i] & 1) { ctx.fillStyle = COLORS.shot; ctx.fillRect(x(i), 4, 2, 12); }
    if (scene.events[i] & 2) { ctx.fillStyle = COLORS.bounce; ctx.fillRect(x(i), 4, 2, 12); }
    if (scene.missing_3d[i]) {
      ctx.fillStyle = scene.event_missing[i] ? "#5484d4" : "#a3bdce";
      ctx.fillRect(x(i), 29, dx, 10);
    }
  }
  ctx.strokeStyle = "#e0eaf5"; ctx.beginPath(); ctx.moveTo(x(frame), 0); ctx.lineTo(x(frame), h); ctx.stroke();
}

export function valuesFor(points, target, missing, mode, fps) {
  if (!points) return null;
  return points.map((point, i) => {
    if (missing?.[i]) return null;
    if (mode === "error") return Math.hypot(...point.map((value, j) => value - target[i][j]));
    if (mode === "speed") return i === 0 || missing?.[i - 1] ? null : Math.hypot(...point.map((value, j) => value - points[i - 1][j])) * fps;
    return point[{ x: 0, y: 1, z: 2 }[mode]] ?? null;
  });
}

export function drawGraph(canvas, scene, frame, mode, visibility) {
  const { ctx, w, h } = surface(canvas);
  const left = 47, right = w - 12, top = 10, bottom = h - 22;
  const target = scene.gt_3d;
  const missing = scene.missing_3d;
  const series = {};
  for (const kind of ["gt", "input", "prediction"]) {
    if (!visibility[kind]) continue;
    const points = scene[`${kind}_3d`];
    series[kind] = valuesFor(points, target, kind === "input" ? missing : null, mode, scene.fps);
  }
  const values = Object.values(series).flatMap((s) => (s || []).filter((v) => v !== null && Number.isFinite(v)));
  let low = values.length ? Math.min(...values) : 0, high = values.length ? Math.max(...values) : 1;
  if (high - low < 1e-6) high = low + 1;
  const pad = (high - low) * 0.07; low = ["error", "speed"].includes(mode) ? 0 : low - pad; high += pad;
  const x = (i) => left + i / Math.max(1, scene.frames - 1) * (right - left);
  const y = (v) => bottom - (v - low) / (high - low) * (bottom - top);
  ctx.fillStyle = "#243d5a";
  missing.forEach((value, i) => { if (value) ctx.fillRect(x(i), top, Math.max(1, (right - left) / scene.frames), bottom - top); });
  for (let tick = 0; tick <= 3; tick++) {
    const v = low + (high - low) * tick / 3;
    ctx.strokeStyle = "#2a3a4d"; ctx.beginPath(); ctx.moveTo(left, y(v)); ctx.lineTo(right, y(v)); ctx.stroke();
    const label = v !== 0 && Math.abs(v) < 0.01 ? v.toExponential(1) : v.toFixed(Math.abs(v) < 10 ? 1 : 0);
    ctx.fillStyle = "#8d9eb4"; ctx.textAlign = "right"; ctx.fillText(label, left - 6, y(v) + 3);
  }
  ctx.textAlign = "left"; ctx.fillText("0 s", left, h - 5); ctx.textAlign = "right";
  ctx.fillText(`${((scene.frames - 1) / scene.fps).toFixed(2)} s`, right, h - 5); ctx.textAlign = "left";
  for (const [kind, values] of Object.entries(series)) {
    if (!values) continue;
    ctx.strokeStyle = COLORS[kind]; ctx.lineWidth = kind === "input" ? 0.9 : 1.5; ctx.globalAlpha = kind === "input" ? 0.6 : 1;
    line(ctx, values, (value, i) => [x(i), y(value)]);
  }
  ctx.globalAlpha = 1; ctx.lineWidth = 1; ctx.strokeStyle = "#d7e5f5";
  ctx.beginPath(); ctx.moveTo(x(frame), top); ctx.lineTo(x(frame), bottom); ctx.stroke();
}

export function frameFromPointer(event, canvas, frames, left = 47) {
  const rect = canvas.getBoundingClientRect();
  return Math.max(0, Math.min(frames - 1, Math.round((event.clientX - rect.left - left) / Math.max(1, rect.width - left - 12) * (frames - 1))));
}

export function drawEvents(canvas, scene, frame) {
  const { ctx, w, h } = surface(canvas), left = 47, right = w - 12, top = 14, bottom = h - 24;
  const x = (i) => left + i / Math.max(1, scene.frames - 1) * (right - left);
  const y = (p) => bottom - p * (bottom - top);
  ctx.fillStyle = "#243d5a";
  scene.missing_3d.forEach((m, i) => { if (m) ctx.fillRect(x(i), top, Math.max(1, (right - left) / scene.frames), bottom - top); });
  for (const p of [0, 0.5, 1]) {
    ctx.fillStyle = "#9aaec4"; ctx.fillText(String(p), 15, y(p) + 3);
    ctx.strokeStyle = "#2a3a4d"; ctx.beginPath(); ctx.moveTo(left, y(p)); ctx.lineTo(right, y(p)); ctx.stroke();
  }
  for (const [values, color] of [[scene.event_target, COLORS.gt], [scene.event_probability, COLORS.prediction]]) {
    if (!values) continue;
    ctx.strokeStyle = color; ctx.lineWidth = 1.8; line(ctx, values, (p, i) => [x(i), y(p)]);
  }
  ctx.strokeStyle = "#e0eaf5"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x(frame), top); ctx.lineTo(x(frame), bottom); ctx.stroke();
  ctx.fillStyle = "#9aaec4"; ctx.fillText("0 s", left, h - 5); ctx.fillText(`${((scene.frames - 1) / scene.fps).toFixed(2)} s`, right - 42, h - 5);
}
