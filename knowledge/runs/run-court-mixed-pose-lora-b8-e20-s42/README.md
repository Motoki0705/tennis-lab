# Prediction bundle

`pred_test.npz` はGit管理用に次の通りcompact化している。

- `kp_heatmaps`と`line_probability`は`float16`。元のargmaxおよび0.5閾値判定と一致することを検証済み。
- `seg_mask`は値域検証後に`uint8`化。元のmaskと完全一致する。
- `seg_logits`と`line_logits`は、それぞれ保存済み`seg_mask`と`line_probability`に対して冗長であり、単一ファイル100MB未満に収めるため除外した。
- pose出力、keypoint座標・score・valid、scene IDは元のdtypeと値を保持している。

学習時に保存された未compact版は、gitignore対象の`.training_queue/repro/1789052279895515108_1993066_court_mixed_pose_lora_b8_e20_s42/predictions/pred_test.npz`に残している。
