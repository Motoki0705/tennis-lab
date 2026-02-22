export type Side = "near" | "far";
export type TargetMode = "none" | "cell" | "point";

export type Vec2 = { x: number; y: number };
export type Vec3 = { x: number; y: number; z: number };

export type CellBounds = {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
};

export type CellInfo = {
  cell_id: number;
  side: Side;
  bounds: CellBounds;
  center: Vec2;
};

export type CellsResponse = { cells: CellInfo[] };

export type CourtGeometryResponse = {
  keypoints: number[][]; // [20][3]
  segments: number[][]; // [[i,j], ...]
};

export type SimulateShotRequest = {
  from_side: Side;
  from_cell: number;
  target_mode: TargetMode;
  to_cell?: number;
  target_point?: Vec2;
  shot?: {
    position?: Vec3;
    velocity?: Vec3;
    spin?: Vec3;
  };
  physics?: {
    use_drag?: boolean;
    use_magnus?: boolean;
  };
  sim?: {
    max_sim_frames?: number;
    sim_fps?: number;
    output_fps?: number;
  };
  seed?: number;
};

export type SimulateShotResponse = {
  positions: number[][];
  velocities: number[][];
  fps_out: number;
  sim_fps: number;
  events: {
    t_net: number;
    t_fence: number;
    t_bounce1: number;
    t_bounce2: number;
    net_pos: Vec3 | null;
    bounce1_pos: Vec3 | null;
    bounce2_pos: Vec3 | null;
  };
  labels: {
    category: string;
    to_cell: number | null;
  };
  metrics: {
    apex_height_m: number;
    time_to_bounce1_s: number | null;
    net_clearance_m: number | null;
  };
};
