import assert from "node:assert/strict";
import test from "node:test";
import {
  COURT_CORNERS,
  resizeFromCorner,
} from "../../../../../src/synthetic_data_generation/alignment/manual/static/resize.mjs";

const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);
function cornerWorld(court, scale, [x, y]) {
  const angle = (court.angle_degrees * Math.PI) / 180;
  return {
    u: court.u + scale * (Math.cos(angle) * x - Math.sin(angle) * y),
    v: court.v + scale * (Math.sin(angle) * x + Math.cos(angle) * y),
  };
}
for (const angle of [0, 37, 90, -143]) {
  for (const [index, corner] of COURT_CORNERS.entries()) {
    test(`opposite corner stays fixed at yaw ${angle}, handle ${index}`, () => {
      const court = { u: 3.2, v: -7.4, angle_degrees: angle },
        scale = 0.7;
      const opposite = corner.map((x) => -x);
      const fixed = cornerWorld(court, scale, opposite);
      const dragged = cornerWorld(court, scale, corner);
      for (const factor of [0.4, 1, 1.6]) {
        const delta = {
          u: (dragged.u - fixed.u) * (factor - 1),
          v: (dragged.v - fixed.v) * (factor - 1),
        };
        const result = resizeFromCorner(court, scale, corner, delta);
        near(result.scale, scale * factor);
        const updated = { ...court, u: result.u, v: result.v };
        const anchor = cornerWorld(updated, result.scale, opposite);
        near(anchor.u, fixed.u);
        near(anchor.v, fixed.v);
        const moved = cornerWorld(updated, result.scale, corner);
        near(moved.u, dragged.u + delta.u);
        near(moved.v, dragged.v + delta.v);
      }
    });
  }
}
test("perpendicular movement does not resize, and crossing the anchor never flips the court", () => {
  const court = { u: 2, v: 3, angle_degrees: 0 },
    corner = COURT_CORNERS[0];
  const sideways = resizeFromCorner(court, 1, corner, {
    u: -corner[1],
    v: corner[0],
  });
  near(sideways.scale, 1);
  near(sideways.u, court.u);
  near(sideways.v, court.v);
  const resized = resizeFromCorner(court, 1, corner, {
    u: -4 * corner[0],
    v: -4 * corner[1],
  });
  assert.ok(resized.scale > 0);
  const before = cornerWorld(
    court,
    1,
    corner.map((x) => -x),
  );
  const after = cornerWorld(
    { ...court, ...resized },
    resized.scale,
    corner.map((x) => -x),
  );
  near(before.u, after.u);
  near(before.v, after.v);
});
