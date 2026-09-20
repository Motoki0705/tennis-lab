// Decode the review wire format without depending on the DOM or Three.js.
// Legacy scenes contain one `entity`; mixed scenes contain `entities`, with
// each group's buffer starting on the next four-byte boundary.

export function entityGroups(scene) {
  const groups = scene.entities ?? [scene.entity];
  if (!groups.length || groups.some((group) => !group || group.frames !== scene.frame_count)) {
    throw new Error("Entity groups must share the scene timeline.");
  }
  return groups;
}

export function decodeBuffers(buffer, scene) {
  let offset = 0;
  const groups = entityGroups(scene).map((entity) => {
    offset = Math.ceil(offset / 4) * 4;
    const values = entity.slots * entity.frames;
    const joints = values * entity.joint_count * 3;
    const size = joints * 4 + (entity.orientation ? values * 8 : 0) + (entity.presence ? values : 0);
    if (offset + size > buffer.byteLength) throw new Error("Entity buffer is truncated.");
    const frames = new Float32Array(buffer, offset, joints);
    offset += joints * 4;
    let orientation = null;
    if (entity.orientation) {
      orientation = new Float32Array(buffer, offset, values * 2);
      offset += values * 8;
    }
    let presence = null;
    if (entity.presence) {
      presence = new Uint8Array(buffer, offset, values);
      offset += values;
    }
    return { entity, frames, presence, orientation };
  });
  if (offset !== buffer.byteLength) throw new Error("Entity buffer has unexpected trailing bytes.");
  return groups;
}

/** Build mixed ball/player entities, including players with only a 3D root. */
export function buildModel(scene, groups) {
  const entities = [];
  for (const [groupIndex, group] of groups.entries()) {
    const { entity, frames: data, presence, orientation } = group;
    const frameCount = entity.frames;
    const jointCount = entity.joint_count;
    for (let slot = 0; slot < entity.slots; slot += 1) {
      const length = frameCount * jointCount * 3;
      const positions = data.subarray(slot * length, (slot + 1) * length);
      const roots = new Float32Array(frameCount * 3);
      for (let frame = 0; frame < frameCount; frame += 1) {
        const base = frame * jointCount * 3;
        for (let axis = 0; axis < 3; axis += 1) {
          roots[frame * 3 + axis] = jointCount === 1
            ? positions[base + axis]
            : (positions[base + 11 * 3 + axis] + positions[base + 12 * 3 + axis]) / 2;
        }
      }
      entities.push({
        id: `group-${groupIndex}-slot-${slot}`,
        color: entity.colors[slot % entity.colors.length],
        kind: entity.kind,
        frames: frameCount,
        joints: jointCount,
        positions,
        roots,
        heading: orientation?.subarray(slot * frameCount * 2, (slot + 1) * frameCount * 2) ?? null,
        presence: presence?.subarray(slot * frameCount, (slot + 1) * frameCount) ?? null,
        edges: entity.skeleton,
      });
    }
  }
  return {
    court: scene.court,
    frames: scene.frame_count,
    entities,
    cameras: scene.cameras.map((camera) => ({
      id: camera.id,
      label: camera.id,
      center: camera.center,
      frustum: camera.frustum,
      rotation: camera.rotation,
    })),
  };
}
