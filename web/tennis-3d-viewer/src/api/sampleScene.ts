import type { TennisSceneDocument } from "@/types/scene";

export const sampleScene: TennisSceneDocument = {
  schema_version: "tennis_scene_v2",
  metadata: {
    scene_id: "sample",
    source: "sample-data",
    experiment_name: "demo",
    created_at: new Date().toISOString(),
  },
  scene: {
    fps: 30,
    num_frames: 60,
    num_cameras: 1,
    cameras: [
      {
        id: "cam_0",
        image_size: [1280, 720],
        camera_C: [0, -12, 5],
        camera_R: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        camera_intr: [800, 640, 360],
      },
    ],
    court: {
      type: "standard",
    },
  },
  tracks: [
    {
      id: "pred_player_0",
      entity_type: "player",
      source: "prediction",
      model_id: "mvpose_v3",
      label: "Player 0",
      color_hint: "#ff8a65",
      frames: Array.from({ length: 60 }, (_, frame_index) => {
        const phase = frame_index / 60;
        return {
          frame_index,
          joints_3d: Array.from({ length: 17 }, (_, jointIdx) => [
            Math.sin(phase * Math.PI * 2) * 1.5,
            jointIdx * 0.05,
            Math.cos(phase * Math.PI * 2) * 1.5 + jointIdx * 0.01,
          ]),
          valid: true,
          scores: {
            exist_conf: 0.95,
          },
        };
      }),
    },
  ],
};
