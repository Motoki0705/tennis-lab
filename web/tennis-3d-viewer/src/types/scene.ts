export interface TennisSceneDocument {
  schema_version: string;
  metadata: SceneMetadata;
  scene: SceneWorld;
  tracks: SceneTrack[];
  overlays_2d?: SceneOverlays2D;
  metrics?: SceneMetrics;
  events?: SceneEvent[];
}

export interface SceneMetadata {
  scene_id: string;
  source: string;
  dataset_split?: string;
  experiment_name?: string;
  created_at?: string;
  notes?: string;
  extra?: Record<string, unknown>;
}

export interface SceneWorld {
  fps: number;
  num_frames: number;
  duration_sec?: number;
  num_cameras: number;
  cameras: SceneCamera[];
  court?: SceneCourt;
}

export interface SceneCamera {
  id: string;
  image_size: [number, number];
  camera_C: Vec3;
  camera_R: Matrix3x3;
  camera_intr: [number, number, number];
  extra?: Record<string, unknown>;
}

export interface SceneCourt {
  type?: string;
  keypoints_3d?: Vec3[];
  meta?: Record<string, unknown>;
}

export interface SceneTrack {
  id: string;
  entity_type: string;
  source: string;
  model_id?: string;
  label?: string;
  color_hint?: string;
  frames: SceneTrackFrame[];
  extra?: Record<string, unknown>;
}

export interface SceneTrackFrame {
  frame_index: number;
  joints_3d?: Vec3[];
  racket_3d?: Vec3[];
  points_3d?: Vec3[];
  valid?: boolean;
  scores?: Record<string, number>;
  extra?: Record<string, unknown>;
}

export interface SceneOverlays2D {
  cameras: Record<string, SceneOverlayCamera>;
}

export interface SceneOverlayCamera {
  frames: SceneOverlayFrame[];
}

export interface SceneOverlayFrame {
  frame_index: number;
  court_keypoints_2d?: PointsWithVisibility;
  players?: SceneOverlayEntity[];
  extra?: Record<string, unknown>;
}

export interface SceneOverlayEntity {
  track_id: string;
  joints_2d?: PointsWithVisibility;
  extra?: Record<string, unknown>;
}

export interface PointsWithVisibility {
  points: Vec2[];
  visibility: number[];
}

export interface SceneMetrics {
  per_track?: Record<string, Record<string, number>>;
  global?: Record<string, Record<string, number>>;
  extra?: Record<string, unknown>;
}

export interface SceneEvent {
  type: string;
  start_frame: number;
  end_frame?: number;
  players_involved?: string[];
  extra?: Record<string, unknown>;
}

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
export type Matrix3x3 = [Vec3, Vec3, Vec3];
