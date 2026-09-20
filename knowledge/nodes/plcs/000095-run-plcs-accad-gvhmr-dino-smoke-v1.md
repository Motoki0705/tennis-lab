---
task: plcs
sequence: 95
recorded_at: 2026-09-14
date_source: experiment_date
papers: [paper-2024-gvhmr]
id: run-plcs-accad-gvhmr-dino-smoke-v1
type: run
title: ACCADとDINO/GVHMR共通COCO17によるPLCS生成試験
provider: codex
date: '2026-09-14'
status: done
config:
  motion_sources: accad_gvhmr
  simulation.num_scenes: 4
  run.num_workers: 1
metrics:
  generated_scenes: 4
  generated_camera_views: 24
  accad_scenes: 2
  gvhmr_scenes: 2
  output_size_mib_approx: 6.7
artifacts:
  output_dir: /home/kamimura/projects/tennis-lab/data/plcs/_smoke/accad_gvhmr_dino_v1
repro:
  branch: feat/plcs-gvhmr-motion-source
  command: /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.plcs.scripts.generate_dataset motion_sources=accad_gvhmr simulation.num_scenes=4 run.num_workers=1 run.output_dir=plcs/_smoke/accad_gvhmr_dino_v1 paths.project_root=/home/kamimura/projects/tennis-lab
parents: []
relations: []
tags:
- plcs
- gvhmr
- dino
- motion-source
---

## 考察 / Findings

### 要約

ACCADとDINO経由で抽出したGVHMRモーションを共通COCO17形式で扱い、実データから4シーン・24カメラビューを生成できた。これは生成経路の動作確認であり、PLCSの予測精度改善を測った学習実験ではない。

### アーキテクチャ詳細

ACCADはSMPL-H、GVHMRは世界座標SMPLパラメータからCOCO17へ変換し、右手系Z-up・メートル・root translation/full rotationを共通契約として使う。混合設定のGVHMR重みは0.5。入力インデックスは実行時点でACCAD167件、GVHMR5件だった。

### メトリクスの解釈

実際に選ばれたsourceはACCAD2シーン、GVHMR2シーン。保存FPSはそれぞれ120と59.94006であり、元の時間間隔を保持した。フレーム数は生成順に1017、217、1010、426。保存容量は約6.7 MiB。データ生成のみなので収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

形式別adapterを通過した双方のモーションを、共通のcourt配置・カメラ投影・writerで処理できた。native FPSが出力にも残っており、生成時の60fps固定変換は行われていない。学習時の28/30/60fps再サンプリングは別途単体・dataset統合テストで検証している。

### 既存実験との比較

このrunでは精度の比較実験を行っていない。ACCAD-onlyに対する改善幅は未評価。

### 次に有効な実験

全114件のGVHMR抽出と品質確認後、同じ固定val/testでACCAD-onlyと混合trainを比較する。大量の固定データ複製を避ける場合はchunk生成のmotion_sources設定を利用できる。
