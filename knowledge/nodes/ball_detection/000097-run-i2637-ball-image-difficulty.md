---
id: run-i2637-ball-image-difficulty
type: run
task: ball_detection
sequence: 97
recorded_at: '2026-11-26'
title: Meiji 4camの球像の画素径・ブラーと外部注釈の来歴
issue: 2637
provider: codex
date: '2026-11-26'
status: done
summary_impact: none
config:
  dataset: meiji_4cam/video_000
  external_labels: clip_000-068_ball_positions
  seed: 42
  resolution_wh: [[1920, 1080], [1280, 720], [512, 288]]
  footprint_samples_per_clip_camera: 8
  visual_review_offsets: [-4, -2, 0, 2, 4]
metrics:
  clips: 69
  camera_frames: 184912
  outside_coordinate_centres: 4136
  footprint_anchors: 2208
  measured_by_resolution: [2121, 2080, 360]
  median_diameter_px: [4.0, 2.9721, 1.7638]
  median_blur_proxy_px: [2.6458, 2.1909, 1.633]
  measured_at_all_sizes: 339
  visual_reviewed: 64
  visual_size_reviewed: 18
  visual_occluded: 2
  visual_unresolved_in_frame: 11
  visual_out_of_frame: 1
repro:
  commit: 365d15e92cc848479d3417a33c295a81fb31e781
  command: >-
    env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    .venv/bin/python knowledge/runs/run-i2637-ball-image-difficulty/measure.py
    --root data/tennis_multiview/processed/meiji_4cam/dataset/videos/video_000
    --output outputs/ball-paper/size --workers 3
artifacts:
  run_dir: knowledge/runs/run-i2637-ball-image-difficulty
parents: []
relations: []
papers: []
tags: []
---

## 考察 / Findings

### 外部座標の来歴

初版69 JSONは、Chromeが2026-11-24 01:50:43 UTCに `chatgpt.com` から取得した記録のある
`clip_000-068_ball_positions.zip` と全件バイト一致した。JSON内の作成時刻は11月23日16:48:44から
11月24日01:09:38 UTC。ZIPと69メンバーのSHA-256を固定した。初版のmetadata、ZIPとダウンロードの記録、
2026-11-23〜25の対応するローカルCodexセッション、注釈campaign記録と #1739/#2458/#2462 を照合した。
初版の生成モデルと実行IDは確認できなかった。ブラウザの記録は取得経路を示す。
[来歴・ハッシュ](../../runs/run-i2637-ball-image-difficulty/provenance.json) ·
[#1739 の初版インポート](https://github.com/Motoki0705/tennis-lab/issues/1739#issuecomment-7120448233)

11月24日20:27 JSTに開始した `campaign-ball-reannot-20261124` では、clip_012、015、026を
Codex CLIの `gpt-6.2-sol / max` が再注釈した。ローカルstateの `events.log`、各 `codex.jsonl` と
セッションの `turn_context` からモデルと日時、終了値0を確認し、campaign入力を初版、3出力を公開rev1と照合した。
初版は69 clipの計測入力で、rev1の採用はこの3 clipのベンチマーク更新に対応する。
[#1739 の再注釈開始](https://github.com/Motoki0705/tennis-lab/issues/1739#issuecomment-7130142039) ·
[#1739 のrev1採用](https://github.com/Motoki0705/tennis-lab/issues/1739#issuecomment-7131999600)

### 画素径とブラー

全69 clip・4視点の元映像1920×1080、約59.94 fpsを確認した。カメラごとに各clipの画面内 `observed` から
seed 42で8フレームを選んだ。前後6 frameを確保できる中心が対象。局所窓で前後3・6 frameの
画素中央値を背景とし、注釈周辺12元画像pxの強い正負の残差から同符号の連結成分を計測した。
実際にグレースケール化した画像をOpenCV `INTER_LINEAR` で1280×720、512×288に縮小して同じ処理を行った。
面積3画素未満、低コントラスト、窓に切れる成分、位置ずれ、過大な成分は欠測として記録した。
短軸分散で円盤相当径を、長短の分散差で一様な線分相当の長さを求めた。後者は画素分布から求めた
見かけの伸びの指標である。

| 画像サイズ | 計測成立 | 径の中央値［P05–P95］px | ブラー指標の中央値［P05–P95］px |
| --- | ---: | ---: | ---: |
| 1920×1080 | 2,121 / 2,208（96.1%） | 4.00［2.31–6.80］ | 2.65［0.60–6.05］ |
| 1280×720 | 2,080 / 2,208（94.2%） | 2.97［1.76–4.62］ | 2.19［0.00–4.24］ |
| 512×288 | 360 / 2,208（16.3%） | 1.76［1.15–2.35］ | 1.63［0.00–5.73］ |

3解像度すべてで計測できた339フレームに限ると径の中央値は5.96、4.09、1.76 px、ブラー指標は
4.19、2.83、1.63 px。512では主に画素不足で欠測が増え、全計測成立例同士の比較には異なる標本が入る。
目視で輪郭の追えた18件の元画像で、短軸の概測中央値は3 px、長軸は4.5 px。短軸に対する自動計測の絶対差の
中央値は0.31 pxだった。観測区分からの固定監査32件では、元画像で計測できた31件のうち26件で球像を確認し、
4件は案内近傍で識別不能、1件は近傍にずれた候補があった。測れた成分に他の像が混ざる例を含む。
[分布図](../../runs/run-i2637-ball-image-difficulty/footprint-distributions.png) ·
[条件・カメラ別集計](../../runs/run-i2637-ball-image-difficulty/footprint-summary.csv) ·
[64フレームの目視記録](../../runs/run-i2637-ball-image-difficulty/visual-review.csv)

### カメラ間の差と見え方

| カメラ | 範囲外の座標中心 / 46,228 | 元画像の径中央値 px | 元画像残差ピーク中央値（輝度0–255） | clip平均ビットレートの中央値 Mbps（コンテナ込み） |
| --- | ---: | ---: | ---: | ---: |
| cam0 | 688（1.49%） | 3.94 | 69.5 | 10.88 |
| cam1 | 630（1.36%） | 3.71 | 57.0 | 8.17 |
| cam2 | 756（1.64%） | 4.20 | 91.25 | 4.32 |
| cam3 | 2,062（4.46%） | 4.19 | 67.5 | 8.70 |

全184,912件中に名目範囲外の座標中心が4,136件（2.24%）、うち `estimated` が4,125件。
外部可視性フラグは可視123,007、`unverified` 57,821、不可視表記3,613、明示 `out_of_frame` 471。
別にcamごとの3層を固定し、32件の画面内観測・16件のその他画面内・16件の範囲外座標を目視した。
64件中、可視27（42.2%）、ずれ候補1、遮蔽2（3.1%）、画面内の識別不能11（17.2%）、
画面外1（1.6%）、判定保留22（34.4%）。遮蔽は球像と人物の前後関係、画面外は縁への移動を確認したもの。
範囲外座標16件のうち15件は退出を視認できなかった。ここでの割合は層化した監査の分母に対応し、
全収録の真の画面外率・遮蔽率・不可視率は未確定。
[可視性フラグの全数](../../runs/run-i2637-ball-image-difficulty/visibility-flags.csv) ·
[監査の層・カメラ別分母](../../runs/run-i2637-ball-image-difficulty/visibility-audit-summary.csv) ·
[図](../../runs/run-i2637-ball-image-difficulty/visibility.png)

球像の小ささ、入力縮小による分離の難しさ、遮蔽と画面外の実例を支持する。
カメラ間で球像・画面外中心率・ビットレートが異なった。画質の因果的な効果は撮影距離、背景・照明、符号化との
分離ができておらず部分的な支持。既存のBall Detection summaryのモデル選定・比較の判断は維持する。

### 実行記録

systemd user job `20261126T113758Z_b215433fb167_ball-paper-footprint` は終了値0、全276映像をデコードし
575.14秒（親CLIのサービス経過576.10秒）。元動画・clip.json・外部JSONの版は
[media.csv](../../runs/run-i2637-ball-image-difficulty/media.csv)にある。目視は実装worker 1名で、対象視点と
必要な他視点の前後画像を利用した。画像チェック用の切り出しはローカル出力の `audit-media/` に残した。

```bash
run=knowledge/runs/run-i2637-ball-image-difficulty
.venv/bin/python -B "$run/summarize.py" --input outputs/ball-paper/size \
  --review "$run/visual-review.csv" --output outputs/ball-paper/size-summary
```
