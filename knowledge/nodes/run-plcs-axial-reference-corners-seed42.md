---
id: run-plcs-axial-reference-corners-seed42
type: run
title: PLCS 四隅3–4入力 reference axial (seed42)
provider: codex
session: 01a08470-1d91-72d3-8da6-13497c8b30d1
date: '2026-09-09'
status: done
config:
  model:
    name: plcs_multiview_axial_reference
    io:
      input_profile: multiview
    hidden_dim: 512
    num_layers: 8
    num_heads: 8
    ffn_dim: 1408
    ffn_type: swiglu
    dropout: 0.1
    invisible_init_std: 0.02
    max_views: 4
    max_seq_len: 256
    rope_dim: 16
    rope_theta_time: 1000.0
    rope_theta_camera: 1000.0
    predict_canonical_pose: true
    canonical_pose_readout: temporal_decomposition
    target_frame_contract: reference_camera_court_rzpi_v1
    axial_rope_contract: time_camera_reference_selector_v1
    reference_selector_mode: reference
  loss:
    position_weight: 1.0
    position_smooth_l1_beta: 0.1
    rotation_weight: 0.1
    angle_weight: 0.1
    position_smoothness_weight: 0.0
    canonical_pose_weight: 1.0
    canonical_pose_smooth_l1_beta: 1.0
    reprojection_weight: 1.0
    reprojection_smooth_l1_beta: 0.01
    joint_angle_weight: 0.0
    torsion_angle_weight: 0.0
    torso_twist_weight: 0.0
    bone_length_weight: 0.0
    joint_angle_velocity_weight: 0.0
    torsion_angle_velocity_weight: 0.0
    torso_twist_velocity_weight: 0.0
    joint_angle_velocity_angle_weights: null
    torsion_angle_velocity_angle_weights: null
  data:
    scene_dir: plcs/single_object_camera_view_v2
    num_court_kp: 14
    camera_candidates:
    - 0
    - 1
    - 2
    - 3
    num_views_range:
    - 3
    - 4
    seq_len_range:
    - 128
    - 128
    batch_size: 4
  training:
    learning_rate: 0.0001
    weight_decay: 0.1
    compile:
      enabled: true
      backend: inductor
      mode: default
      fullgraph: false
      dynamic: false
    trainer:
      max_epochs: 50
      gradient_clip_val: 1.0
      deterministic: true
      precision: bf16-mixed
      log_every_n_steps: 50
      check_val_every_n_epoch: 1
      accumulate_grad_batches: 1
      reload_dataloaders_every_n_epochs: 0
      enable_progress_bar: true
      enable_model_summary: true
      benchmark: false
  seed: 42
metrics:
  loss: 0.008255734108388424
  position_error_m: 0.13123983144760132
  angular_error_deg: 7.78125
  position_accuracy_0.5m: 0.9713019728660583
  angle_accuracy_15deg: 0.8649799823760986
  canonical_mpjpe_m: 0.0984630212187767
  canonical_pck_0.1m: 0.6652390360832214
repro:
  commit: 0dafcdee375981846d1bc34835d780937941935b
  branch: feat/plcs-axial-reference
  remote: https://github.com/Motoki0705/tennis-lab.git
  command: .venv/bin/python -m src.tasks.plcs.scripts.train --config-name train_axial_reference
  commit_scope: architecture snapshot from checkpoint metadata; exact launch commit
    unavailable
artifacts:
  run_dir: knowledge/runs/run-plcs-axial-reference-corners-seed42
  output_dir: /home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-axial-reference/outputs/plcs/axial_reference_corners_v3-4_t128
  tb_logdir: .claude/worktrees/plcs-axial-reference/outputs/plcs/axial_reference_corners_v3-4_t128/logs/version_0
  checkpoint: /home/kamimura/projects/tennis-lab/ckpt/plcs/plcs-axial-reference-corners-v3-4-t128-seed42-epoch47.ckpt
  curves: knowledge/runs/run-plcs-axial-reference-corners-seed42/curves.png
parents: []
tags:
- plcs
- axial
- reference
- camera-view-v2
- kp14
- seed42
---

## 考察 / Findings

### 要約
四隅から3–4ビューを入力する reference axial PLCS は50 epochsを完了。validation位置誤差で選択したepoch47は0.107815 m。記録されたtestは最終epoch49の値であり、best47のtest値ではない。

### アーキテクチャ詳細
hidden512・8層・8heads・SwiGLU FFN1408。時間／カメラ軸RoPEとreference selectorで、指定カメラ側のcourt Rzπ座標系へ位置・yaw・canonical poseを出力する。canonical poseはTemporalDecomposedCanonicalPoseHead。camera_view_v2の14 court KPと人物17 KPを128フレーム入力し、BS4、seed42、bf16、compile有効。回転／angle重み各0.1、position／canonical pose／reprojection重み各1。完全な実行設定は同runのconfig.yamlに保存。

### メトリクスの解釈
frontmatterはepoch49のtest: 位置0.131240 m、角度7.78125度、canonical MPJPE0.098463 m、PCK@0.1m=0.665239。best47のvalidation位置0.107815 mと最終49のvalidation位置0.111780 mは異なるチェックポイントの評価。test後処理はrunner.pyが最終in-memory modelを評価するため、bestへの自動復元を仮定しない。
収束曲線は原TensorBoard値から生成。終盤46–49 epochのvalidation位置は0.112307→0.107815→0.112360→0.111780 mで上下しており、位置の最良epochと総lossの最小epochは一致しない。

曲線の目視では約25,000 stepsでlossと角度誤差が大きく低下し、その後もcanonical MPJPEと位置誤差が漸減する。原因を特定する証拠はないのでこの変化を特定の設計効果と断定しない。終盤にvalidationの持続的悪化は見られない。

### アーキテクチャ⇄メトリクスの因果考察
reference座標と近遠KP意味を揃えた学習がsynthetic validation/testで成立したことは観測できる。ただしhead分解やreprojection単独の寄与を分離した対照実験はなく、改善原因の断定はできない。仮説: 実動画では2D pose誤差・遮蔽・合成データとの差がcanonical poseの誤差を増やす。

### 既存実験との比較
同じ分割／reference契約で直接比較できる登録済みbaselineは確認できなかったため、因果的な比較edgeは作らない。PLCSとBLCSは出力対象が異なるため数値順位を比較しない。

### 次に有効な実験
best47そのもののtestを別途評価し、最終49と区別して追記する。実Meiji clipで2D観測・reference規約・3D姿勢を目視と再投影で検証する。test予測配列は既存runが未保存のため、この登録だけでは新指標を再計算できない。再現用commitはmetadata上のarchitecture snapshotで、厳密なlaunch commitは未回収。
