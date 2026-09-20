---
id: run-b00-clay-flare-50-nht-30k-v1
type: run
title: B00 Flare 50枚・NHT 30,000ステップ
provider: codex
session: 01a0bd70-e78d-7732-ba19-1a4595e91a32
date: '2026-09-20'
status: done
config:
  model: NHT
  data: B00 Flare clay 50 views
  max_steps: 30000
  seed: 42
  data_factor: 2
  cap_max: 1000000
  pose_opt: false
  train_images: 42
  validation_images: 8
  initialization: from scratch
metrics:
  psnr: 23.13494110107422
  ssim: 0.6246808767318726
  lpips: 0.23034197092056274
  lpips_alex: 0.23034197092056274
  cc_psnr: 22.767940521240234
  cc_ssim: 0.606555700302124
  cc_lpips: 0.24095302820205688
  ellipse_time: 0.012073665857315063
  num_GS: 1000000
  elapsed_seconds: 1983.055991094
  common_validation_psnr: 23.13494110107422
  common_validation_ssim: 0.6246808767318726
  common_validation_lpips: 0.23034197092056274
repro:
  commit: 893d0ca4129e9e69f8e8c850d89c6200cf2df2e3
  branch: codex/b00-clay-variant
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.claude/worktrees/b00-clay-variant/.venv/bin/python
    -m src.synthetic_data_generation.scripts.run_appearance_variant action=execute_training
    variant.output_root=/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-50-30k-v001
artifacts:
  run_dir: knowledge/runs/run-b00-clay-flare-50-nht-30k-v1
  output_dir: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-50-30k-v001/reconstruction/3dgs/model
  log: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-50-30k-v001/reconstruction/logs/nht_training/attempt-1.log
  checkpoint: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-50-30k-v001/reconstruction/3dgs/model/ckpts/ckpt_29999_rank0.pt
  export: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-50-30k-v001/reconstruction/export/scene.json
  comparison: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/comparison-flare-50-7k-30k-v001/index.html
  curves: knowledge/runs/run-b00-clay-flare-50-nht-30k-v1/curves.png
  tb_logdir: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-50-30k-v001/reconstruction/3dgs/model/tb
  verification: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-50-30k-v001/review/verification.json
  code_provenance: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-50-30k-v001/provenance/code
parents:
- run-b00-clay-flare-50-v1
relations:
- to: run-b00-clay-flare-nht-7k-v1
  rel: compares
tags:
- synthetic-data
- nht
- clay
- 30k
---

## 考察 / Findings

### 要約
既存の生成50枚をそのまま使い、NHTをゼロから30,000ステップ学習して標準exportまで完了した。PSNR・LPIPSは7k版より改善した一方、SSIMは低下し、白線が薄く途切れる視点が見られた。

### アーキテクチャ詳細
学習42枚・評価8枚、元SfM全491カメラによる正規化、seed 42、factor 2、pose_opt=false、cap 100万を維持。7k checkpointからの継続ではなく新規初期化。max_stepsに依存するLRや最後の3,000ステップのcolor refinementスケジュールも30k用になるため、単純な同一scheduleの延長ではない。

### メトリクスの解釈
同じ8評価視点でPSNR 23.1349 dB（7k:22.7179）、SSIM 0.6247（7k:0.6527）、LPIPS 0.2303（7k:0.2625）。trainer subprocess時間は1,983.06秒、約33.1分。checkpoint内step=29999、100万Gaussianと全splat値の有限性、50カメラのexport、座標変換の同一性、元データのハッシュを確認。指標はNHT標準評価であり、既存trainerでは保存checkpointから1回optimizerを更新した後の評価となる。

### アーキテクチャ⇄メトリクスの因果考察
8視点の比較で、7kの地面にあった大きな筋状のにじみは弱まり、クレーの細かい質感が増えた。一方、frame 0/40/152/192では白線が薄くなったり断続的になる。96/208の近いネットにもぼけや一部消失が残る。画像全体のPSNR・LPIPS改善だけではコート線・ネットの保存品質を判断できない。

### 既存実験との比較
評価用入力のPNG画素ハッシュが7k版と一致することを照合した。7k→30kで全指標が一律に改善したわけではなく、細い構造の保持について長時間学習だけで解決したとは言えない。

### 次に有効な実験
追加50枚を含む100枚版の30k学習を後続queueで実行する。共通する8評価視点で白線・ネットと指標を比較し、枚数増加の効果を確認する。
