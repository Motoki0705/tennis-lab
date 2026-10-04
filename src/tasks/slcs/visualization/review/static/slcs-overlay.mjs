// Saved normalized UV uses division by image width/height, matching SceneResult.
export const PLAYER_COLORS = ["#3a6fb0", "#d1623f"];

export function poseCrop(points, width, height) {
  const visible = points.filter((p) => p && p.every(Number.isFinite) && p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1);
  if (!visible.length) return null;
  const xs = visible.map((p) => p[0] * width), ys = visible.map((p) => p[1] * height);
  const h = Math.min(height, Math.max(150, (Math.max(...ys) - Math.min(...ys)) * 1.8));
  const w = Math.min(width, h * 1.6);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  return [Math.max(0, Math.min(width - w, cx - w / 2)), Math.max(0, Math.min(height - h, cy - h / 2)), w, h];
}

export function ballResidualPixels(ball, width, height) {
  if (!ball.uv || !ball.projection.uv) return null;
  return Math.hypot((ball.uv[0] - ball.projection.uv[0]) * width, (ball.uv[1] - ball.projection.uv[1]) * height);
}

export function drawOverlay(ctx, sample, skeleton, options) {
  const [width, height] = sample.image_size;
  const point = (uv) => [uv[0] * width, uv[1] * height];
  const dot = (uv, color, radius, ring = false) => {
    if (!uv) return;
    const [x, y] = point(uv);
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.strokeStyle = color; ctx.lineWidth = 3;
    if (ring) ctx.stroke(); else ctx.fill();
  };
  if (options.court) for (const uv of sample.court.uv) dot(uv, "#40e0bf", 5);
  if (options.pose) for (const player of sample.players) {
    ctx.strokeStyle = PLAYER_COLORS[player.slot]; ctx.lineWidth = 3;
    for (const [a, b] of skeleton) {
      if (!player.pose_uv[a] || !player.pose_uv[b]) continue;
      ctx.beginPath(); ctx.moveTo(...point(player.pose_uv[a])); ctx.lineTo(...point(player.pose_uv[b])); ctx.stroke();
    }
    for (const uv of player.pose_uv) dot(uv, PLAYER_COLORS[player.slot], 4);
  }
  if (options.ball) dot(sample.ball.uv, "#ffe84a", 5);
  if (options.projection) {
    for (const player of sample.players) dot(player.projection.uv, "#bb67f1", 10, true);
    dot(sample.ball.projection.uv, "#bb67f1", 9, true);
  }
}
