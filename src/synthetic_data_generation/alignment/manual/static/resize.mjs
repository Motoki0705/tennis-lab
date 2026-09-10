/** Regulation court corners in the editor's right-handed local UV frame. */
export const COURT_CORNERS = [
  [-5.485, -11.885],
  [5.485, -11.885],
  [5.485, 11.885],
  [-5.485, 11.885],
];

/** Resize along the court diagonal, keeping the opposite corner in world space.
 * delta is pointer motion since pointer-down, so grabbing a handle off-centre
 * never introduces a jump. Perpendicular motion cannot distort the aspect ratio.
 */
export function resizeFromCorner(court, scale, corner, delta) {
  const angle = (court.angle_degrees * Math.PI) / 180;
  const x = Math.cos(angle) * corner[0] - Math.sin(angle) * corner[1];
  const y = Math.sin(angle) * corner[0] + Math.cos(angle) * corner[1];
  const nextScale = Math.max(
    1e-6,
    scale + (delta.u * x + delta.v * y) / (2 * (x * x + y * y)),
  );
  return {
    scale: nextScale,
    u: court.u + x * (nextScale - scale),
    v: court.v + y * (nextScale - scale),
  };
}
