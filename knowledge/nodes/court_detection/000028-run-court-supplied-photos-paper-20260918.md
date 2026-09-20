---
id: run-court-supplied-photos-paper-20260918
type: run
task: court_detection
sequence: 28
recorded_at: '2026-09-18'
title: 指定写真4枚のCourt推論とB00–B03アライメントの論文化
provider: codex
date: '2026-09-18'
status: done
session: 01a0b44d-2642-7040-8df3-0cad69ce7cd8
config:
  model: multiscale-depth3-epoch17-vs-tcd-e5cd4f1
  loss: inference_only
  data: supplied_tennis_court_four_photos
  device: cpu
metrics:
  external_images: 4
  baseline_official_homography_count: 0
  baseline_argmax_homography_count: 1
  ours_homography_count: 4
  corpus_images_audited: 17256
  exact_duplicate_count: 0
  rendered_alignment_views: 12
artifacts:
  output_dir: paper/court_robustness/evidence
  report: paper/court_robustness/report.pdf
  predictions: paper/court_robustness/evidence/predictions
parents: []
relations: []
tags: [court-detection, qualitative, cpu-inference, 3dgs]
---

## 考察 / Findings

### 要約
ユーザー指定画像を全数推論し、実レンダリングに保存アライメントを重ねたB00〜B03各3例とともに論文化した。結果の詳細・解釈は `artifacts.report` を正本とする。

### アーキテクチャ詳細
保存重みと混合学習設定を変更せず利用した。構成・損失・前処理は論文第3・4節、設定とハッシュは同梱evidenceを参照する。

### メトリクスの解釈
H生成数は計算できた件数であり、正しいコートの件数ではない。人手GTなしのため精度や有意差は算出しない。完全一致の非検出も、事前学習や会場の非重複を保証しない。

### アーキテクチャ⇄メトリクスの因果考察
LINEとKP由来Hの不整合を観測した。整合損失は無効だが、それを原因とする主張にはアブレーションが必要である。

### 既存実験との比較
TCDは公式処理に加えてargmax感度確認を掲載した。B01・B02の保存アライメントは人手確認を含むため、全自動の成果として扱わない。

### 次に有効な実験
会場分離・人手GT付きの評価と、モデル・解像度を固定した実写のみ／合成混合の比較。
