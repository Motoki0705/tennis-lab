export type VideoMeta = {
  fps: number;
  frame_count: number;
  width: number;
  height: number;
};

export type BallClipConfig = {
  start_frame: number;
  clip_length: number;
};

export type BallFrameAnnotation = {
  visibility: 0 | 1 | 2;
  x_px: number;
  y_px: number;
  score: number;
  source: "manual" | "assist" | "unknown";
};

export type CourtKeypoint = {
  x_px: number;
  y_px: number;
  visibility: 0 | 1;
  source: "manual" | "assist" | "homography" | "unknown";
};

export type CourtFrameAnnotation = {
  frame_idx: number;
  keypoints: CourtKeypoint[];
};

export type ExportResult = { output_dir: string };

export type BallAssistMeta = {
  checkpoint_path: string | null;
  model_type: "wasb" | "hrcnet";
  device: "cpu" | "cuda";
  batch_size: number;
  score_threshold: number;
  max_disp: number;
  created_at: string;
};

export type BallAssistSummary = {
  available: boolean;
  clip_matches_current: boolean;
  clip: BallClipConfig | null;
  meta: BallAssistMeta | null;
  count: number;
};

export type BallAssistRunResult = {
  clip: BallClipConfig;
  meta: BallAssistMeta;
  count: number;
};

export type BallAssistAll = {
  annotations: Record<string, BallFrameAnnotation>;
};
