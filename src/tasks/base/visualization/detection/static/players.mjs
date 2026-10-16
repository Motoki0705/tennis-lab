// COCO-17 in stored-image pixels. Scores are raw model peaks, not probabilities.
export const COCO_EDGES = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

export function playerColor(id) {
  let hash = 0;
  for (const char of String(id)) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return `hsl(${(hash * 137.508) % 360} 90% 68%)`;
}

export function drawPlayers(ctx, people, options, scale) {
  if (!options.players) return;
  ctx.save();
  ctx.lineCap = ctx.lineJoin = "round";
  for (const person of people) {
    const [x1, y1, x2, y2] = person.box;
    const height = Math.max(0, (y2 - y1) * scale);
    const line = Math.max(0.45, Math.min(1.1, height / 160));
    const details = height >= 55;
    const color = playerColor(person.id);
    ctx.strokeStyle = ctx.fillStyle = color;
    ctx.lineWidth = Math.max(0.6, line) / scale;
    if (options.trails && person.trail.length > 1) {
      ctx.globalAlpha = 0.65;
      ctx.beginPath();
      person.trail.forEach(([x, y], i) =>
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y),
      );
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.lineWidth = Math.max(0.5, Math.min(1, line)) / scale;
    if (options.boxes) ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (options.pose) {
      ctx.lineWidth = line / scale;
      ctx.beginPath();
      for (const [a, b] of COCO_EDGES) {
        // At a distance, facial links and joint disks cover the body. Keep
        // the limb geometry; zooming restores the full COCO-17 detail.
        if (!details && a < 5 && b < 5) continue;
        ctx.moveTo(...person.keypoints[a].slice(0, 2));
        ctx.lineTo(...person.keypoints[b].slice(0, 2));
      }
      ctx.stroke();
      if (details) {
        const radius = Math.max(0.55, Math.min(1, height / 220)) / scale;
        ctx.beginPath();
        for (const [x, y] of person.keypoints) {
          ctx.moveTo(x + radius, y);
          ctx.arc(x, y, radius, 0, 2 * Math.PI);
        }
        ctx.fill();
      }
    }
    if (options.identities) {
      const label = person.id.startsWith("raw_")
        ? `R${person.raw_track_id}`
        : `${person.id.replace(/^player_(\d+)$/, "P$1")} / R${person.raw_track_id}`;
      const fontSize = Math.max(7, Math.min(9, height / 10));
      ctx.font = `${fontSize / scale}px system-ui`;
      const y = Math.max((fontSize + 2) / scale, y1 - 3 / scale);
      ctx.fillStyle = "#10251fe6";
      ctx.fillRect(
        x1 - 1 / scale,
        y - (fontSize + 1) / scale,
        ctx.measureText(label).width + 2 / scale,
        (fontSize + 3) / scale,
      );
      ctx.fillStyle = color;
      ctx.fillText(label, x1, y);
    }
  }
  ctx.restore();
}
