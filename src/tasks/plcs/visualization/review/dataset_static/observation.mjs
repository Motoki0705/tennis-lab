// Dependency-free observation decoding and per-frame dataset diagnostics.

export function decodeObservations(buffer, document) {
  if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
    throw new Error("Observation buffers require a little-endian platform.");
  }
  if (buffer.byteLength !== document.byte_length) throw new Error("Observation buffer size mismatch.");
  const fields = {};
  let expectedOffset = 0;
  for (const [name, field] of Object.entries(document.buffer_fields)) {
    const count = field.shape.reduce((total, size) => total * size, 1);
    if (field.dtype !== "float32" || field.byte_offset !== expectedOffset ||
        !field.shape.every(size => Number.isInteger(size) && size > 0)) {
      throw new Error(`Invalid observation field: ${name}`);
    }
    fields[name] = new Float32Array(buffer, field.byte_offset, count);
    expectedOffset += count * 4;
  }
  if (expectedOffset !== buffer.byteLength) throw new Error("Observation buffer has trailing bytes.");
  return fields;
}

export function frameObservation(fields, cameraIndex, frame, kind) {
  const count = kind === "human" ? 17 : 20;
  const prefix = `cam_${cameraIndex}_${kind}_`;
  const uv = fields[prefix + "uv"].subarray(frame * count * 2, (frame + 1) * count * 2);
  const visible = fields[prefix + "vis"].subarray(frame * count, (frame + 1) * count);
  const projected = kind === "human"
    ? fields[prefix + "projected"].subarray(frame * count * 2, (frame + 1) * count * 2)
    : fields[prefix + "projected"];
  const front = kind === "human"
    ? fields[prefix + "front"].subarray(frame * count, (frame + 1) * count)
    : fields[prefix + "front"];
  return { count, uv, visible, projected, front };
}

export function inImage(u, v) {
  return u >= 0 && u <= 1 && v >= 0 && v <= 1;
}

export function frameStatistics(observation, imageSize, pointCount = observation.count) {
  let visible = 0, mismatch = 0, outside = 0, compared = 0, squaredError = 0, maxError = 0;
  for (let index = 0; index < pointCount; index += 1) {
    const u = observation.uv[index * 2], v = observation.uv[index * 2 + 1];
    const pu = observation.projected[index * 2], pv = observation.projected[index * 2 + 1];
    const isVisible = observation.visible[index] === 1;
    const expected = observation.front[index] === 1 && inImage(pu, pv);
    visible += Number(isVisible);
    mismatch += Number(isVisible !== expected);
    outside += Number(isVisible && !inImage(u, v));
    if (isVisible && observation.front[index] === 1) {
      const error = Math.hypot((u - pu) * imageSize[0], (v - pv) * imageSize[1]);
      squaredError += error * error;
      maxError = Math.max(maxError, error);
      compared += 1;
    }
  }
  return { visible, mismatch, outside, compared, maxError: compared ? maxError : null,
    rmsError: compared ? Math.sqrt(squaredError / compared) : null };
}

export function playerCrop(observation, imageSize) {
  const points = [];
  for (let index = 0; index < observation.count; index += 1) {
    if (observation.visible[index] === 1) {
      points.push([observation.uv[index * 2] * imageSize[0], observation.uv[index * 2 + 1] * imageSize[1]]);
    }
  }
  if (!points.length) return null;
  const left = Math.min(...points.map(point => point[0]));
  const top = Math.min(...points.map(point => point[1]));
  const right = Math.max(...points.map(point => point[0]));
  const bottom = Math.max(...points.map(point => point[1]));
  const size = Math.max(50, (right - left) * 1.5, (bottom - top) * 1.5);
  return [(left + right - size) / 2, (top + bottom - size) / 2, size, size];
}
