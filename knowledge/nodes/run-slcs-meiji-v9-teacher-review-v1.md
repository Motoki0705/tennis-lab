---
id: run-slcs-meiji-v9-teacher-review-v1
type: run
title: 'Meiji v9教師のRGB重畳確認: train/valの3clipとunsupported区間'
provider: codex
date: '2026-09-19'
status: done
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
config:
  data: slcs/meiji_rgb_v9
  clips: [video_000/clip_000, video_000/clip_009, video_001/clip_000]
  device: cpu
metrics:
  reviewed_clips: 3
  selected_frames: 30
  rgb_panels: 90
  positive_ball_frames: 2730
  positive_player_frames: 6077
repro:
  commit: f4a887a9
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python -m scripts.analysis.render_reconstruction_review
    --dataset-root /home/kamimura/projects/tennis-lab/data/slcs/meiji_rgb_v9
    --run-root /home/kamimura/projects/tennis-lab/outputs/tennis_scene/generate/meiji_rgb_v9/s42-002
    --output-root /home/kamimura/projects/tennis-lab/outputs
    --output tennis_scene/visualize/meiji_teacher_review/s42-takeover-001
    --clip video_000/clip_000 --clip video_000/clip_009 --clip video_001/clip_000
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-teacher-review-v1
  output_dir: outputs/tennis_scene/visualize/meiji_teacher_review/s42-takeover-001
parents: [run-slcs-meiji-v9-clip009-repair-v1]
relations: []
tags: [slcs, meiji, quality, visualization, cpu]
---

## 考察 / Findings

### 要約

train収録2clip、val収録1clipのraw/refined教師をRGBに重ね、3枚のcontact sheetと3枚の軌跡図を確認した。
選択frameでは補正後のrootが人物の腰位置に概ね一致し、特にcam2のrawの系統的ずれが軽減した。
unsupported区間の不連続は残るため、0 weightを橙色で表示し、見た目の改善だけで全点を採用しない。

### アーキテクチャ詳細

1clipにつき等間隔5点に加え、同じ正weight mask上のraw/refined最大ball残差、最小支持、
最大root補正、最大root frame差分を選ぶ（今回は各10frame）。映像3視点と生の2D観測、
raw/refined 3Dの再投影を表示。XYZ・実FPSの速度・学習weightも別図に保存した。
由来と観測配列の一致を表示前に検査し、rawとrefinedのmaskは共通とした。

### メトリクスの解釈

3clip計3,070frame中ball正weightは2,730frame、player-slot正weightは6,077/6,140。
この件数はframe選択前の全時系列であり、学習windowの採用件数とは異なる。
目視確認は30時点・90画像の限定的な確認で、独立3D精度・全clip・test収録の保証ではない。
新規学習がないため収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

clip009の約8.9秒やval clipの欠測区間には大きな速度spikeがあり、0 weightが隣接する。
補正後rootの腰への一致はhip/shoulder支持付き幾何補正と整合的だが、これだけで奥行きやyawを
検証したとはいえない。平均再投影誤差とともに、支持率・欠測長・連続支持点の速度を併記する。

### 既存実験との比較

数値監査にRGB重畳を追加した。clip009修復の品質監査と矛盾する明らかなslot入替えは選択frameでは
見られなかった。raw/refinedの一方だけを表示した以前の確認から、同じ画像上の比較へ拡張した。

### 次に有効な実験

video_002の生成完了後に同じ選択方針で追加確認する。教師の学習maskと評価maskを維持し、
SLCSの定数予測問題は平均値baseline・軌跡分散・速度で独立に検証する。
