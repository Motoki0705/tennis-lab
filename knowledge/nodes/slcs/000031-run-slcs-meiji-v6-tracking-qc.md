---
task: slcs
sequence: 31
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-meiji-v6-tracking-qc
type: run
title: 'Meiji v6全体生成: 人物対応の切替を確認し再生成へ'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: failed
config:
  model: selected Meiji PLCS/BLCS 60epoch
  data: meiji_rgb_v6; broadcast_rgb_v4
  people_selection: largest per court half
metrics:
  available_scene_caches_at_cancel: 15
  inspected_pose_receipts: 47
  stale_pose_receipts: 1
  inspected_rgb_feature_receipts: 56
  stale_rgb_feature_receipts: 3
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'env TENNIS_RGB_GPU=0 bash /home/kamimura/projects/tennis-lab/.claude/worktrees/slcs-real-rgb/scripts/datasets/build_real_rgb.sh
    --execute all '
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v6-tracking-qc
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v6/s42-001
  log: .training_queue/logs/1789727504849046017_215604_slcs-real-rgb-build-all.log
  diagnostics: knowledge/runs/run-slcs-meiji-v6-tracking-qc/partial_quality_report.json
parents:
- run-slcs-plcs-meiji-real-final-eval
- run-slcs-meiji-observation-mask-probe-v1
- run-slcs-meiji-rgb-features-v2
relations: []
tags:
- slcs
- meiji
- quality-control
- person-association
- cancelled
---

## 考察 / Findings

### 要約
Meiji全体生成中の品質確認で、半面ごとの最大box選択が別人へ切り替わる実例を確認した。共有queueの取消・process group終了を確認し、v6を採用済みデータとはしない。完全性検査や支持maskだけでは人物対応の正しさを保証できない。

### アーキテクチャ詳細
指定Court checkpoint、outsourceボール、選定済みMeiji PLCS/BLCS、幅6.5mの人物範囲、長欠損maskを使用。DINO検出後に各frame独立の最大boxを選ぶ旧方式。途中の成功cacheは維持し、新たな人物関連付けを別の観測・dataset版で検証する。

### メトリクスの解釈
available_scene_caches_at_cancelは取消時にdatasetに存在するscene数で、先行probeの成果も含む。このjob単独の新規生成数ではない。partial_quality_report.jsonは12clip時点の途中snapshotである。未生成と明示除外を含み、全56clip完了を意味しない。

video_000/clip_007ではcam1のsample576以降に手前の人物、cam0の576/580と620以降に別人物を選ぶ。本来の選手はraw detectionに残る。P0教師の81frame欠損は品質maskで抑制されるが、人物の2D入力自体を直す必要がある。video_001/clip_000の選択画像でも旧方式が手前の通行人を選ぶ。

加えて47件のpose receiptのうち001/001 cam2だけViTPose checkpoint SHAが異なり、buildが拒否した。56件のRGB feature receiptのうち001/007・002/014・002/016も現在のcheckpoint SHAと異なる。原因は特定できておらず、重み本体の変更や手編集と断定しない。receiptを書き換えて一致扱いせず、3clipのRGB特徴は新datasetで再計算する。

### アーキテクチャ⇄メトリクスの因果考察
人物対応の切替は映像とraw候補・選択IDから確認できる。コート範囲へ入る大きな通行人が選ばれるため、3Dモデルの再学習だけでは2D対応の誤りを解消できない。一方000/000の教師欠損区間では元選手が選ばれており、全欠損をtracking問題とはしない。

### 既存実験との比較
先行mask probeは長い非観測を扱えたが、別人が検出される場合のID switchは防げなかった。支持付き再投影平均の改善も全clip・全視点の改善ではなく、000/005ではball p95が悪化した。詳細はmanual_review_notes.jsonに観測と制約を分けて保存した。

### 次に有効な実験
明示的な時間的関連付けで同一人物を維持し、raw DINOとCourtの検証済みcacheを再利用してViTPose/教師を再生成する。旧データの3D/人物cacheを新方式と混在させない。初期選択と画面外からの復帰も実映像で確認した後、全Meiji生成・品質集計を完了する。cache_reuse_receipt.jsonとseed_shared_caches.pyは任意の移行手順であり、fresh入力からの通常buildには不要。
