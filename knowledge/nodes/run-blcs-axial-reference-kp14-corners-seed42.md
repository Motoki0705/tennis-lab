---
id: run-blcs-axial-reference-kp14-corners-seed42
type: run
title: BLCS KP14・Colab L4・四隅3–4入力 reference axial (seed42)
provider: codex
session: 01a08470-1d91-72d3-8da6-13497c8b30d1
date: '2026-09-09'
status: done
config:
  model:
    name: blcs_multiview_axial_reference
    io:
      input_profile: multiview
    hidden_dim: 512
    num_layers: 8
    num_heads: 8
    camera_layers_per_stage:
    - 1
    - 1
    - 1
    - 1
    - 1
    - 1
    - 1
    - 1
    time_layers_per_stage:
    - 1
    - 1
    - 1
    - 1
    - 1
    - 1
    - 1
    - 1
    time_global_stage_mask:
    - false
    - false
    - false
    - false
    - false
    - false
    - false
    - false
    attention_type: mha
    num_kv_heads: null
    ffn_dim: 1408
    ffn_type: swiglu
    dropout: 0.1
    rope_dim: 16
    rope_theta_time: 10000.0
    rope_theta_camera: 1000.0
    time_window_radius: 16
    max_seq_len: 256
    max_num_cameras: 4
    predict_velocity: false
    invisible_init_std: 0.02
    num_court_tokens: 14
    target_frame_contract: reference_camera_court_rzpi_v1
    axial_rope_contract: time_camera_reference_selector_v1
    reference_selector_mode: reference
  loss:
    position_weight: 1.0
    position_axis_weights: null
    reprojection_weight: 1.0
    smoothness_weight: 0.0
    gravity_weight: 0.0
    smoothness_order: 3
    smoothness_beta: 0.001
    gravity_beta: 0.005
    smoothness_axis_weights: null
  data:
    scene_dir: blcs/single_object_camera_view_v2
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
    batch_size: 16
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
      log_every_n_steps: 100
      check_val_every_n_epoch: 1
      accumulate_grad_batches: 1
      reload_dataloaders_every_n_epochs: 0
      enable_progress_bar: true
      enable_model_summary: true
      benchmark: false
  seed: 42
metrics:
  loss: 0.00019348060595802963
  position_error_m: 0.22284646332263947
  position_accuracy_0.3m: 0.8301770687103271
  endpoint_error_m: 0.6715567111968994
repro:
  commit: 41da7f0aedbad30bf0cc65852b2ee979b9e05d11
  branch: tmp/blcs-axial-reference-colab
  remote: https://github.com/Motoki0705/tennis-lab.git
  command: bash scripts/colab/run.sh run blcs_axial_reference_kp14 --gpu L4 --drive-mode
    mount --download-to outputs/colab_downloads --keep-on-failure
  commit_scope: exact Colab source revision
artifacts:
  run_dir: knowledge/runs/run-blcs-axial-reference-kp14-corners-seed42
  output_dir: /home/kamimura/projects/tennis-lab/.claude/worktrees/tmp-blcs-axial-reference-colab/outputs/colab_restart/kp14_completed/outputs/colab/blcs_axial_reference_kp14
  tb_logdir: .claude/worktrees/tmp-blcs-axial-reference-colab/outputs/colab_restart/kp14_completed/outputs/colab/blcs_axial_reference_kp14/logs/version_0
  checkpoint: /home/kamimura/projects/tennis-lab/ckpt/blcs/blcs-axial-reference-kp14-corners-v3-4-t128-seed42-best.ckpt
  curves: knowledge/runs/run-blcs-axial-reference-kp14-corners-seed42/curves.png
parents: []
tags:
- blcs
- axial
- reference
- camera-view-v2
- kp14
- seed42
---

## 考察 / Findings

### 要約
camera_view_v2データをKP14で読み出すBLCSをColab L4で新規学習し、50 epochs／20,900 steps完了。best=最終epoch49でvalidation位置誤差0.225221 m、test位置誤差0.222846 m。旧KP20重みを初期化・resumeに使用していない。

### アーキテクチャ詳細
hidden512・8stages・8heads、各stageでcamera/time attentionを各1層、time window radius16、SwiGLU FFN1408。court token14、ball UV、visibility、reference selectorから指定カメラのcourt Rzπ座標系へ3D軌道を予測。四隅候補[0,1,2,3]から3–4ビュー、128フレーム、BS16、seed42、bf16、compile有効。positionとreprojectionの重み各1、smoothness／gravity重み0。v2データのdisk上court20点をreference整列してから14点に切り出す契約であり、disk20はKP20入力学習を意味しない。完全な実行設定はconfig.yaml。

### メトリクスの解釈
test位置誤差0.222846 m、0.3m以内の割合0.830177、endpoint誤差0.671557 m。best checkpointが最終epoch49と一致するのでこのtest値は採用重みの値として扱える。終盤46–49 epochのvalidation位置は0.228188→0.235736→0.227219→0.225221 mで、bestは最後に更新された。総平均よりendpoint誤差が大きいため、時間窓端の品質を確認する価値がある。ただしこの指標だけでは全ての境界に同じ問題があるとは断定できない。

曲線の目視では前半のvalidation変動が大きいが、後半は位置・endpoint誤差とも低下し、0.3m accuracyが上昇している。50 epochs末でもvalidation改善が残り、長期学習による頭打ちは未確認。trainとvalには大きな差があり、augmentationや入力条件の差が影響する可能性はあるが、この曲線だけから一般化の優劣を断定しない。

### アーキテクチャ⇄メトリクスの因果考察
KP14入力でsynthetic validation/testが成立したことが観測事実。仮説: 有限の時間窓と端点周辺の少ない文脈がendpoint誤差の一因である。外部ballの補間・遮蔽推定を含む実動画は合成データと誤差分布が異なり、synthetic精度から実動画精度を保証できない。KP数以外の効果を分離する対照実験は実施していない。

### 既存実験との比較
ユーザー指定により今回のKP14 runだけを登録する。旧KP20 runのノードや比較実験は追加しない。PLCSとは出力対象が異なるため直接の性能比較は行わない。

### 次に有効な実験
Meiji clipの外部ball観測をそのまま入力し、窓端の不連続、ネット通過、バウンド、遮蔽区間を目視と再投影で点検する。まずデータ・座標契約の整合を確認し、その後必要なら時間窓端の集計を追加する。test予測配列は元runで未保存のため未登録であり、追加指標算出には再推論が必要。
