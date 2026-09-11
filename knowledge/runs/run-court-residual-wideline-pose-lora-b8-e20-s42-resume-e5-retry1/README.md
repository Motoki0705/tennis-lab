# Prediction bundle

`pred_test.npz`はGit管理用に次のとおりcompact化している。

- `kp_heatmaps`と`line_probability`は`float16`。元の空間argmaxおよび0.5閾値判定と一致することを検証済み。
- `seg_mask`は値域と完全一致を検証して`uint8`化した。
- `seg_logits`と`line_logits`は、保存済み`seg_mask`と`line_probability`に対して冗長なため除外した。
- pose出力、keypoint座標・score・valid、scene IDは元のdtypeと値を保持している。

学習時の未compact版は、gitignore対象の`.training_queue/repro/1789131317875449519_3473120_court_residual_wideline_pose_lora_b8_e20_s42_resume_e5_retry1/predictions/pred_test.npz`に保持している。
