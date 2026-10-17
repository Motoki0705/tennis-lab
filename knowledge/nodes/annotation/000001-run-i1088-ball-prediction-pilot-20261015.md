---
id: run-i1088-ball-prediction-pilot-20261015
type: run
task: annotation
sequence: 1
recorded_at: '2026-10-15'
title: 新規実クリップのMDD予測を共通GPU工程で完走
issue: 1088
provider: codex
session: 01a13898-67ee-7641-86c1-4cf77195d15c
date: '2026-10-15'
status: done
config:
  campaign: initial-model-preview-20261015
  clip_id: 269ec7212d2f6cfec639ef83b47af4a2d577ff29299d1ca0e1e2ab6bfafb073e
  inference:
    coordinates: native_pixel_endpoint
    decode: pyav_rgb24_native
    device: cuda
    overlap_policy: nearest_window_center_then_earliest_start
    precision: bfloat16
    schema_version: ball.annotation-inference.v1
    stride: 16
    window_frames: 32
  model_version: mdd-convnext-v2-s42-epoch009
  weights_sha256: 3c250d313b745d6c6430ce544c01f8a630eb146439251b4b83f3028829f22f5d
metrics:
  media_frames: 898
  annotation_target_frames: 778
  stage_elapsed_seconds: 87.32537991000027
repro:
  commit: d734b1308ed9181e7ed2ffb8d3d8c941f747f590
  branch: codex/annotation-budget-roots
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.claude/worktrees/annotation-budget-roots/.venv/bin/python
    -m src.tasks.base.annotation.execution --launch /home/kamimura/projects/tennis-lab/outputs/chat_annotation/campaigns/initial-model-preview-20261015/jobs/f4b10febc079ea6141736acb663e7849eec9abd715064f37a63e8906f6b8c208-1/launch.json
    --sha256 fbe3552aeae9a2fe8d62e4864c1cc73e82788da883488cd0b1105a47544201b4 --queued
artifacts:
  run_dir: knowledge/runs/run-i1088-ball-prediction-pilot-20261015
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1792034617563178564_178010_annotation-fbe3552aeae9a2fe.log
  evidence: knowledge/runs/run-i1088-ball-prediction-pilot-20261015/evidence
parents: []
relations: []
papers: []
tags:
- annotation
- ball_detection
- gpu
- pipeline-smoke
- prediction-only
---

## 観測

2026-10-15、新規YouTube動画 `gy627cn_1rs` から正式な選別・準備経路で作った1clipを、
初期MDD候補の固定重みで推論した。共有GPU queueの排他予約、予算scope、工程supervisorを通り、
exit code 0、工程status succeeded、OperationResult completeと予測receiptが揃った。
保存した座標、入力manifest、設定、終了記録のbytes/hashはbundle内の `evidence/files.json` に残した。

元の15秒窓は641〜656秒。実mediaは1920×1080、60000/1001 fps、898frameで、
注釈対象は先頭/末尾の文脈を除いたframe 60〜837の778frame。
32frameに満たない素材への補完は行っていない。MDDの固定32frame窓と実PTSを使い、
今回のclip生成が通常の入力長を満たすことを確認した。

工程の経過時間は87.325秒で、入力照合・モデル初期化・復号・推論・保存を含む。
queue待機や取得・選別・clip準備の時間は含まない。学習時間・純粋なGPU計算時間・
定常的な処理速度の測定値へ読み替えない。GPU telemetryにはWSLの `[N/A]` により
値が取得できないsampleがあるため、この記録では所有VRAMのpeakを断定しない。

## 解釈と限界

これは1clipの実GPU経路が保存まで完走したという機能確認である。正解注釈との比較は
未実施で、座標精度・レビュー省力化・全動画での安定性は示さない。
未レビュー予測を教師データに採用せず、モデル重みの更新も行っていない。
ball/playerの追加学習2サイクル、10/25/50並列、追跡・poseまでの公開は未実施。
従来の精度・モデル採用判断を変更する結果ではない。

次は同じclipのplayer予測を回収し、両者の描画結果を確認してレビュー範囲・採用条件を
ユーザーと合意する。その後、採用教師の蓄積と追加学習を独立したrunとして記録する。
学習を行っていないため、loss曲線やTensorBoardの学習曲線は存在しない。

## 再現条件

queueが捕捉したcommit・コマンド・差分は改変せず保存した。入力動画・clip・重みの大容量実体は
Gitへ複製していない。再実行にはreceiptが示す固定root上の実体とSHA-256の一致が必要であり、
このbundleだけでの実行を保証しない。既存scope/baselineと累計GPU予算は継承した。

## 抽出画像での初期確認

同じ固定runtimeの `prediction_preview` をCPU工程で実行し、frame
60/140/220/300/380/460/540/620/700/780/837の全画面を1ページに描画して実際に閲覧した。
画像のSHA-256は `a72a619cdd64b3c15dac83d3917efb4e41de8327e0bd5b1ca6c7e2d5947e1169`。
render request/receiptを本bundleへ保存した。元動画由来の画像本体はlocal annotation rootに保持する。

抽出画像ではサーブからラリーへ進む画面に予測点が描画され、frame 140・300などでは
球像の近くにある。一方frame 780などでは球像と点のずれも見える。
縮小した全画面11枚の観察なので、誤差の定量化や全778対象frameの正しさは判定していない。
予測だけを無条件採用する根拠にはせず、連続frame・native cropの追加確認とplayerの結果を
揃えてレビュー方針を決める。これは初期候補の確認であり、採用後に別QA工程を設けるものではない。
