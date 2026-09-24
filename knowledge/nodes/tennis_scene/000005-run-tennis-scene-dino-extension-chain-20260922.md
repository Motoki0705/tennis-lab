---
id: run-tennis-scene-dino-extension-chain-20260922
type: run
task: tennis_scene
sequence: 5
recorded_at: '2026-09-23'
title: 現行Torch用DINO拡張の再buildと32frame GVHMR chain確認
provider: codex
session: 01a0c8c3-d190-7c51-b671-ded223340a2f
date: '2026-09-23'
status: done
config:
  torch: 2.13.0+cu130
  cuda_toolkit: 13.0.88
  arch: sm_120
  build_target: all
  source_policy: existing build.py patches generated sources; upstream and old binary
    unchanged
  frames: 32
  camera: cam0
metrics:
  extension: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib/MultiScaleDeformableAttention.so
  torch: 2.13.0+cu130
  device: NVIDIA GeForce RTX 5060 Ti
  forward_max_abs_difference: 2.384185791015625e-07
  backward_max_abs_differences:
  - 2.384185791015625e-07
  - 2.384185791015625e-07
  - 2.384185791015625e-07
  chain:
    seconds: 42.39212903000043
    arrays:
      smpl_body_pose:
        shape:
        - 2
        - 32
        - 63
        finite: true
      smpl_global_orient:
        shape:
        - 2
        - 32
        - 3
        finite: true
      smpl_betas:
        shape:
        - 2
        - 10
        finite: true
      smpl_vertices_local:
        shape:
        - 2
        - 32
        - 6890
        - 3
        finite: true
      human_kp_2d:
        shape:
        - 2
        - 32
        - 17
        - 2
        finite: true
      human_kp_vis:
        shape:
        - 2
        - 32
        - 17
        finite: true
      bbx_xys:
        shape:
        - 2
        - 32
        - 3
        finite: true
      track_ids:
        shape:
        - 2
        finite: true
      smpl_transl_incam:
        shape:
        - 2
        - 32
        - 3
        finite: true
      smpl_transl_world:
        shape:
        - 2
        - 32
        - 3
        finite: true
      smpl_global_orient_world:
        shape:
        - 2
        - 32
        - 3
        finite: true
repro:
  commit: fa8798a4c19f7efc6b123f4819130d57fe14b565
  branch: codex/tennis-scene-responsibility-cleanup
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup:/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup/.venv/bin/python
    /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_gvhmr_gpu_preflight.py
artifacts:
  run_dir: knowledge/runs/run-tennis-scene-dino-extension-chain-20260922
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790104562109950397_877659_tennis-scene-cleanup-dino-gvhmr-preflight-20260922T1915Z.log
parents:
- run-tennis-scene-meiji-court-regions-dino-block-20260922
relations: []
papers: []
tags:
- dino_extension
- compatibility
- gpu_smoke
---

## 原因と修復

使用していた2026-08-20作成の拡張には、`Tensor.type().is_cuda()`が残っていた。
現行PyTorchのlegacy dispatch変換でPythonDispatcherを扱えず、kernelの実行前に失敗した。
現行repoの正規build処理には`scalar_type()`と`is_cuda()`への互換修正が既にあり、
sourceを専用cacheへコピーしてから適用する。submoduleと既存binaryを変更せず、別のbuild-lib/build-tempへ構築した。

## 検証

GPU処理は共有FIFO queueのresource=allで実行した。新artifactの絶対pathをassertし、
tiny MSDAのforward/backwardを同じ上流のPyTorch参照実装と比較した。最大絶対差はforwardと3種類のgradientのいずれも2.384185791015625e-7。
その後、対象clipのcam0冒頭を32frameへ切り出した動画で、DINO・BoT-SORT・ViTPose・HMR2・GVHMR・SMPL頂点生成を実行した。
2選手×32frame、6890頂点を含む全11配列のshapeと有限性を確認した。chain処理は約42.4秒。

## 限界と次の確認

これは数値kernel互換性と短いモデルchainの接続確認で、人物対応や実クリップ3D精度の評価ではない。
動画はCPUで冒頭32frameを再エンコードした診断入力で、最終評価では元の全1010frame動画を使う。
使用したbinaryの位置・buildコマンド・設定・数値はevidenceへ保存した。TensorBoardは学習を行っていないため対象外。
最終2入口は新しいプロセスで修復済み拡張を指定し、旧Court結果も入力にせず全段を再生成する。
