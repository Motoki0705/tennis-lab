/** Pure frame/visibility and image-plane geometry helpers for BLCS inspection. */
export function finitePoint(point) {
  return Array.isArray(point) && point.length === 2 && point.every(Number.isFinite);
}

export function observationAt(camera, frame) {
  const ball = camera.ball;
  const uv = ball.saved_uv?.[frame] ?? null;
  const projected = ball.projected_uv?.[frame] ?? null;
  const visible = ball.saved_visibility?.[frame] ?? null;
  const expected = ball.expected_visibility[frame];
  let state = visible === true ? "visible" : visible === false ? "not_visible" : "unknown_visibility";
  if (ball.saved_uv === null) state = "missing_uv";
  else if (!finitePoint(uv)) state = "invalid_uv";
  const location = !ball.in_front[frame] ? "behind_camera" : expected ? "in_frame" : "out_of_frame";
  return {uv, projected, visible, expected, state, location,
    mismatch: typeof visible === "boolean" && visible !== expected,
    errorPx: ball.error_px?.[frame] ?? null};
}

/** Extend beyond the sensor to retain finite, non-visible saved observations. */
export function imageTransform(camera, observation, width, height) {
  const [imageWidth, imageHeight] = camera.image_size;
  let left = 0, right = imageWidth, top = 0, bottom = imageHeight;
  for (const point of [observation.uv, observation.projected, ...(camera.court?.saved_uv ?? []), ...(camera.court?.projected_uv ?? [])]) {
    if (!finitePoint(point)) continue;
    left = Math.min(left, point[0] * imageWidth);
    right = Math.max(right, point[0] * imageWidth);
    top = Math.min(top, point[1] * imageHeight);
    bottom = Math.max(bottom, point[1] * imageHeight);
  }
  const margin = 12;
  const scale = Math.min((width - 2 * margin) / (right - left), (height - 2 * margin) / (bottom - top));
  const offsetX = (width - (right - left) * scale) / 2 - left * scale;
  const offsetY = (height - (bottom - top) * scale) / 2 - top * scale;
  return {map: ([u, v]) => [offsetX + u * imageWidth * scale, offsetY + v * imageHeight * scale],
    expanded: left < 0 || top < 0 || right > imageWidth || bottom > imageHeight};
}

/** Candidate continuations stay inspectable, but are absent from active markers. */
export function eventsAt(events, frame) {
  return (events ?? []).filter(event => event.frame === frame && event.status === "on_trajectory");
}
