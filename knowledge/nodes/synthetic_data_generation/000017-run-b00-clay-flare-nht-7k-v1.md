---
id: run-b00-clay-flare-nht-7k-v1
type: run
task: synthetic_data_generation
sequence: 17
recorded_at: '2026-09-20'
title: B00 クレー50枚 Flare → NHT 7,000ステップ
provider: codex
session: 01a0bd70-e78d-7732-ba19-1a4595e91a32
date: '2026-09-20'
status: done
config:
  model: NHT
  data: B00 clay Flare 50 views
  max_steps: 7000
  seed: 42
  data_factor: 2
  pose_opt: false
  cap_max: 1000000
  train_images: 42
  validation_images: 8
  source_cameras: 491
metrics:
  psnr: 22.717906951904297
  ssim: 0.6527017951011658
  lpips: 0.2625459134578705
  lpips_alex: 0.2625459134578705
  cc_psnr: 22.547405242919922
  cc_ssim: 0.6385965347290039
  cc_lpips: 0.26750048995018005
  ellipse_time: 0.011619985103607178
  num_GS: 1000000
  training_steps: 7000
  elapsed_seconds: 384.89981622000005
  camera_count: 50
  scene_scale: 1.5407659983226785
  source_transform_max_abs_difference: 0.0
repro:
  commit: 893d0ca4129e9e69f8e8c850d89c6200cf2df2e3
  branch: codex/b00-clay-variant
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.claude/worktrees/b00-clay-variant/.venv/bin/python
    -m src.synthetic_data_generation.scripts.run_appearance_variant action=execute_training
    variant.output_root=/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001
artifacts:
  run_dir: knowledge/runs/run-b00-clay-flare-nht-7k-v1
  output_dir: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/reconstruction/3dgs/model
  log: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/reconstruction/logs/nht_training/attempt-3.log
  checkpoint: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/reconstruction/3dgs/model/ckpts/ckpt_6999_rank0.pt
  export: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/reconstruction/export/scene.json
  comparison: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/review/nht-validation.html
  comparison_preview: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/review/nht-validation-01.jpg
  code_provenance: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/provenance/code
  verification: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/review/verification.json
  curves: knowledge/runs/run-b00-clay-flare-nht-7k-v1/curves.png
  tb_logdir: data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/reconstruction/3dgs/model/tb
parents:
- run-b00-clay-flare-50-v1
relations:
- to: run-b00-clay-flare-nht-7k-interrupted-v1
  rel: supersedes
papers: []
tags:
- synthetic-data
- nht
- clay
---

## 考察 / Findings

### 要約
Flareで生成した50視点（新規API 49枚＋同条件の比較結果1枚）から、NHTの7,000ステップ学習と標準exportまで完了した。クレーの外観は再現できるが、白線のにじみ・二重化、近いネットのぼけ、地面の不自然な筋状の模様が残った。

### アーキテクチャ詳細
元B00のSfM全491カメラ・217,407点をコピーし、全カメラで座標正規化とscene scaleを計算してからRGB読み込みを50枚に限定した。元の画像順位による分割を保って学習42枚／評価8枚。seed 42、data factor 2、pose_opt=false、最大100万Gaussian。API原本は1536×864、保存JPGは1920×1080、学習準備PNGは960×540。NHT標準の歪み補正後は959×539であり、生成結果を位置合わせするための切り抜きではない。

### メトリクスの解釈
評価8枚の平均PSNRは22.7179 dB、SSIMは0.6527、LPIPSは0.2625。対象もAI生成画像なので実世界の幾何誤差を表す指標ではない。学習・評価のプロセス時間は約384.9秒。checkpoint内のstep=6999と100万GaussianをCPUで読み出して確認した。50カメラのexport、8比較レンダリング、元B00と入力ハッシュ、元シーンとの座標変換の完全一致を検証できた。

### アーキテクチャ⇄メトリクスの因果考察
評価8視点を入力と並べて目視確認した。コートと建物の大きな配置・赤土色は保たれる一方、0/40/56/192/248では白線が不均一になり、96/208では近いネット上端の白帯に広いにじみ・二重化が見える。152でもネットや白線の細部は入力より甘い。地面の筋状の模様や樹木のぼけもある。各画像を独立に編集した際の視点間不整合は原因候補だが、50枚への間引き・短い学習・元再構成の誤差も混在し、このrunだけで原因を断定できない。

### 既存実験との比較
前回はホスト再起動により6,820ステップのログを残して中断し、checkpointは未保存だった。今回は同じ入力でゼロから再実行して完了した。生成APIの追加呼び出しはない。未変換の元画像50枚を同じ学習条件で再学習した比較対照は今回実行していない。

### 次に有効な実験
元画像50枚・同じSfM・同じ7,000ステップを対照として学習し、間引きと画像編集の影響を分離する。その差分を見て、コート面に編集範囲を制限する方法や視点間整合性を高める方法を比較する。
