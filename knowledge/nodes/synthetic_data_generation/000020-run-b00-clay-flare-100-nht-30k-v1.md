---
id: run-b00-clay-flare-100-nht-30k-v1
type: run
task: synthetic_data_generation
sequence: 20
recorded_at: '2026-09-20'
title: B00 Flare 100枚・NHT 30,000ステップ
provider: codex
session: 01a0bd70-e78d-7732-ba19-1a4595e91a32
date: '2026-09-20'
status: done
config:
  model: NHT
  data: B00 Flare clay 100 views
  max_steps: 30000
  seed: 42
  data_factor: 2
  cap_max: 1000000
  pose_opt: false
  train_images: 86
  validation_images: 14
  initialization: from scratch
metrics:
  psnr: 24.347225189208984
  ssim: 0.6698395013809204
  lpips: 0.1710643470287323
  lpips_alex: 0.1710643470287323
  cc_psnr: 24.105770111083984
  cc_ssim: 0.6604482531547546
  cc_lpips: 0.17463041841983795
  ellipse_time: 0.014580930982317244
  num_GS: 1000000
  elapsed_seconds: 2194.7078131910002
  common_validation_psnr: 24.247726678848267
  common_validation_ssim: 0.6587958782911301
  common_validation_lpips: 0.1808630023151636
repro:
  commit: 893d0ca4129e9e69f8e8c850d89c6200cf2df2e3
  branch: codex/b00-clay-variant
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.claude/worktrees/b00-clay-variant/.venv/bin/python
    -m src.synthetic_data_generation.scripts.run_appearance_variant action=execute_training
    variant.output_root=/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-100-30k-v001
artifacts:
  run_dir: knowledge/runs/run-b00-clay-flare-100-nht-30k-v1
  output_dir: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-100-30k-v001/reconstruction/3dgs/model
  log: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-100-30k-v001/reconstruction/logs/nht_training/attempt-1.log
  checkpoint: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-100-30k-v001/reconstruction/3dgs/model/ckpts/ckpt_29999_rank0.pt
  export: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-100-30k-v001/reconstruction/export/scene.json
  comparison: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/comparison-flare-50-100-30k-v001/index.html
  curves: knowledge/runs/run-b00-clay-flare-100-nht-30k-v1/curves.png
  tb_logdir: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-100-30k-v001/reconstruction/3dgs/model/tb
  verification: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-100-30k-v001/review/verification.json
  code_provenance: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-100-30k-v001/provenance/code
parents:
- run-b00-clay-flare-100-v1
relations:
- to: run-b00-clay-flare-50-nht-30k-v1
  rel: compares
papers: []
tags:
- synthetic-data
- nht
- clay
- 30k
---

## 考察 / Findings

### 要約
Flare生成100枚でNHTをゼロから30,000ステップ学習し、checkpoint・評価14視点・標準exportを保存した。共通8視点では50枚30k版よりPSNR・SSIM・LPIPSがすべて改善したが、ネットのぼけと白線の部分的な欠けは残る。

### アーキテクチャ詳細
既存50枚と追加50枚を使用し、学習86枚／評価14枚。元の全491 SfMカメラから正規化・scene scaleを計算し、RGB入力のみ100枚に限定した。seed 42、factor 2、pose_opt=false、cap 100万を50枚版と共有する。追加画像のAPI生成は50枚版の学習と並行し、GPU学習は同じ共有queueで直列実行した。

### メトリクスの解釈
評価14枚の平均はPSNR 24.3472 dB、SSIM 0.6698、LPIPS 0.1711。共通8枚に限ると24.2477 dB、0.6588、0.1809で、50枚30k版の23.1349 dB、0.6247、0.2303を改善した。共通8枚の評価用PNGが同じ画素ハッシュであることを検証しており、評価対象の違いによる平均値の混同を避けた。trainer subprocess時間は2,194.71秒、約36.6分。

### アーキテクチャ⇄メトリクスの因果考察
共通8視点と追加評価6視点を目視した。50枚30k版で薄くなった白線は0/40/152/192などで見えやすくなり、背景と地面の見え方も改善している。一方、96/208や追加評価32では近いネット上端の白帯ににじみやぼけが残る。白線の太さ・連続性も入力と一致せず、240では手前の線の欠けが目立つ。枚数増加による改善は認められるが、生成画像の視点間不整合と再構成の限界を解決したとは言えない。

### 既存実験との比較
生成画像、SfM、正規化、カメラ姿勢、学習パラメータを保存し、元B00のハッシュ不変、scene transformの完全一致、scene scale=1.5407659983を確認した。checkpoint内step=29999、100万Gaussian、全splat値の有限性、100カメラのexport、14枚の比較画像とper-image指標を確認した。指標はNHT標準評価であり、既存trainerでは保存checkpointから1回optimizerを更新した後の評価となる。

### 次に有効な実験
元画像100枚の同条件学習を比較対照にし、画像変換に由来する誤差を切り分ける。白線とネットを元画像から維持する編集範囲の制約や、視点間整合性を高める方式を次の比較候補とする。
