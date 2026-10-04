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
  for (const person of people) {
    const color = playerColor(person.id);
    ctx.strokeStyle = ctx.fillStyle = color;
    ctx.lineWidth = 1.5 / scale;
    if (options.trails && person.trail.length > 1) {
      ctx.globalAlpha = 0.65;
      ctx.beginPath();
      person.trail.forEach(([x, y], i) =>
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y),
      );
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    const [x1, y1, x2, y2] = person.box;
    if (options.boxes) ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (options.pose) {
      ctx.beginPath();
      for (const [a, b] of COCO_EDGES) {
        ctx.moveTo(...person.keypoints[a].slice(0, 2));
        ctx.lineTo(...person.keypoints[b].slice(0, 2));
      }
      ctx.stroke();
      ctx.beginPath();
      for (const [x, y] of person.keypoints) {
        ctx.moveTo(x + 1.8 / scale, y);
        ctx.arc(x, y, 1.8 / scale, 0, 2 * Math.PI);
      }
      ctx.fill();
    }
    if (options.identities) {
      const label = person.id.startsWith("raw_")
        ? person.id
        : `${person.id} · raw ${person.raw_track_id}`;
      ctx.font = `${11 / scale}px system-ui`;
      const y = Math.max(13 / scale, y1 - 5 / scale);
      ctx.fillStyle = "#10251fe6";
      ctx.fillRect(
        x1 - 2 / scale,
        y - 12 / scale,
        ctx.measureText(label).width + 4 / scale,
        15 / scale,
      );
      ctx.fillStyle = color;
      ctx.fillText(label, x1, y);
    }
  }
}
