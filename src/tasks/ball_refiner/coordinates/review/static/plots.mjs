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

export function draw2D(canvas, scene, camera, frame, visibility, only = null) {
  const { ctx, w, h } = surface(canvas);
  const scale = Math.min((w - 20) / 1280, (h - 36) / 720);
  const left = (w - 1280 * scale) / 2, top = (h - 720 * scale) / 2;
  const project = ([x, y]) => [left + x * scale, top + y * scale];
  ctx.fillStyle = "#142d32"; ctx.fillRect(left, top, 1280 * scale, 720 * scale);
  ctx.strokeStyle = "#38515b"; ctx.strokeRect(left, top, 1280 * scale, 720 * scale);
  ctx.save(); ctx.beginPath(); ctx.rect(left, top, 1280 * scale, 720 * scale); ctx.clip();
  ctx.strokeStyle = "#688889";
  for (const [a, b] of scene.court.edges) {
    const points = [scene.court_2d[camera][a], scene.court_2d[camera][b]];
    if (points.every(Boolean)) line(ctx, points, project);
  }
  for (const kind of ["input", "gt", "prediction"]) {
    if (!visibility[kind] || (only && only !== kind)) continue;
    const points = scene[`${kind}_2d`]?.[camera];
    if (!points) continue;
    const valid = (i) => kind !== "input" || !scene.missing_2d[camera][i];
    ctx.strokeStyle = COLORS[kind]; ctx.globalAlpha = kind === "input" ? 0.27 : 0.62;
    ctx.lineWidth = kind === "input" ? 0.8 : 1.5;
    line(ctx, points, project, valid);
    if (kind === "input") {
      ctx.fillStyle = COLORS.input;
      points.forEach((point, i) => { if (valid(i)) { const [x, y] = project(point); ctx.fillRect(x - 1, y - 1, 2, 2); } });
    }
    ctx.globalAlpha = 1;
    if (valid(frame)) {
      const [x, y] = project(points[frame]);
      ctx.fillStyle = COLORS[kind]; ctx.strokeStyle = "#0b1521"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, kind === "gt" ? 5 : 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
  }
  if (!only && visibility.input && !scene.missing_2d[camera][frame]) {
    ctx.strokeStyle = "#bdc8d9"; ctx.setLineDash([3, 3]); ctx.globalAlpha = 0.55;
    line(ctx, [scene.gt_2d[camera][frame], scene.input_2d[camera][frame]], project);
  }
  ctx.restore(); ctx.globalAlpha = 1;
  ctx.fillStyle = "#9aaec4";
  let status = scene.missing_2d[camera][frame] ? "入力: 欠損" : "入力: 観測あり";
  if (!scene.missing_2d[camera][frame]) {
    const [x, y] = scene.input_2d[camera][frame];
    if (x < 0 || x > 1279 || y < 0 || y > 719) status = "入力: 画面外（時系列グラフで確認）";
  }
  ctx.fillText(only === "prediction" && !scene.prediction_2d ? "推論結果なし" : status, 12, h - 7);
}

export function drawTimeline(canvas, scene, camera, frame) {
  const { ctx, w, h } = surface(canvas), left = 47, right = w - 12;
  const x = (i) => left + i / Math.max(1, scene.frames - 1) * (right - left);
  const dx = Math.max(1, (right - left) / scene.frames);
  ctx.fillStyle = "#899fb6"; ctx.fillText("Event", 8, 15); ctx.fillText("2D", 8, 33); ctx.fillText("3D", 8, 50);
  for (let i = 0; i < scene.frames; i++) {
    if (scene.events[i] & 1) { ctx.fillStyle = COLORS.shot; ctx.fillRect(x(i), 4, 2, 12); }
    if (scene.events[i] & 2) { ctx.fillStyle = COLORS.bounce; ctx.fillRect(x(i), 4, 2, 12); }
    if (scene.event_missing[i]) { ctx.fillStyle = "#5484d4"; ctx.fillRect(x(i), 24, dx, 8); }
    else if (scene.isolated_missing[camera][i]) { ctx.fillStyle = "#a3bdce"; ctx.fillRect(x(i), 24, dx, 8); }
    if (scene.missing_3d[i]) { ctx.fillStyle = "#628cc8"; ctx.fillRect(x(i), 41, dx, 8); }
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

export function drawGraph(canvas, scene, camera, frame, dimension, mode, visibility) {
  const { ctx, w, h } = surface(canvas);
  const left = 47, right = w - 12, top = 10, bottom = h - 22;
  const target = dimension === 2 ? scene.gt_2d[camera] : scene.gt_3d;
  if (dimension === 2 && mode === "z") { ctx.fillStyle = "#8d9eb4"; ctx.fillText("Z座標は3Dのみ", left, h / 2); return; }
  const missing = dimension === 2 ? scene.missing_2d[camera] : scene.missing_3d;
  const series = {};
  for (const kind of ["gt", "input", "prediction"]) {
    if (!visibility[kind]) continue;
    const points = dimension === 2 ? scene[`${kind}_2d`]?.[camera] : scene[`${kind}_3d`];
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
