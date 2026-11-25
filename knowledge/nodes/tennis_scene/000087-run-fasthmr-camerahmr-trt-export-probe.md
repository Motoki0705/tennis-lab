---
id: run-fasthmr-camerahmr-trt-export-probe
type: run
task: tennis_scene
sequence: 87
recorded_at: '2026-11-25'
title: FastHMR CameraHMRのTRT16書き出し試行
issue: 2491
provider: codex
session: 01a20d45-575e-7a53-90c5-48b09053fa2b
date: '2026-11-25'
status: failed
config:
  model: fasthmr-camerahmr
  clip: video_000/clip_003
  source_frame: 0
  batch: 1
  image_size: [256, 256]
  tensorrt: 10.14.1.48.post1
metrics:
  export_paths_failed: 2
  engines_built: 0
summary_impact: none
repro:
  commit: 97cabfef23bdab3e1097af32a2cdd96cf754b9dd
  branch: codex/parent-2492-issue-2491-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: sha256sum -c outputs/fasthmr-setup/evaluation/source.sha256 && bash outputs/fasthmr-setup/evaluation/run.sh
    trt-export
artifacts:
  run_dir: knowledge/runs/run-fasthmr-camerahmr-trt-export-probe
  log: .training_queue/logs/0-8204434740687790252_2026-11-25T00-07-39-312209747Z_2393387_i2491-fasthmr-trt16-export.log
parents: [run-fasthmr-camerahmr-smoke]
relations: []
papers: []
tags: [fasthmr, body_recovery]
---

## 考察 / Findings

固定コホートのclip_003・cam0・source frame 0で、実重みと実cropを使った。試行した入力はbatch 1・RGB 256×256、陽性マスクは9,273 pixel。実行HEADは `97cabfef23bdab3e1097af32a2cdd96cf754b9dd`、GPUはRTX 5060 Ti、PyTorch 2.13.0+cu130、CUDA 13.0。 [実行記録](../../runs/run-fasthmr-camerahmr-trt-export-probe/run.json)

両書き出し経路はこの試行で実行不能だった。`torch.export` はcuDNNのTF32設定を演算名なしで問い合わせた時点で失敗し、TorchScriptはLightningのCameraHMRがTrainerに接続されていないという例外で停止した。ONNXとTensorRTエンジンの生成前に終了した。キューの正常終了は診断JSONの保存を表す。 [変換ログの要約](../../runs/run-fasthmr-camerahmr-trt-export-probe/trt-export.json)

新しいPythonプロセスのCPU照合で、同じcuDNNの例外を再現した。旧APIの `torch.backends.cudnn.allow_tf32 = False` だけを使うと照会結果は `False` だった。書き出し専用プロセスのTF32設定を揃えることで最初の例外を避けられる見込みがある。TorchScriptの例外は推論に必要な子層だけを持つ `nn.Module` で迂回できる可能性がある。両案の書き出しは再実行していない。

実cropでマスクを0にするとeagerの最大絶対差は特徴0.625、crop camera 0.07693、betas 0.80396だった。Mask-ToMeの分岐を入力に応じて保持する必要がある。この値は変換診断で、PT32とのparity判定には使わない。標準pipelineの判断は維持する。 [固定版の分岐](https://github.com/TaatiTeam/FastHMR/blob/dc5c8daf0beea56a419c631f494ef7bd7ce35d3c/model/hmr2_tools/patch_vit.py)
