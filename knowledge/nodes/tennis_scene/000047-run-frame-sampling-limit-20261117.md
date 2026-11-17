---
id: run-frame-sampling-limit-20261117
type: run
task: tennis_scene
sequence: 47
recorded_at: '2026-11-17'
title: 1055 frameのCPU実測とframe_sampling上限2048への変更
issue: 1748
provider: codex
date: '2026-11-17'
status: done
summary_impact: updated
config:
  source_commit: 486aa1ab09be857909d0fcf8c2b616c75a53f5b0
  clip_id: video_000/clip_035
  source_frames: 2108
  sampled_frames: 1055
  compared_caps: [2048, 4096]
  timed_repeats: 5
  device: cpu
metrics:
  court_side_mean_seconds: 0.05531220460543409
  camera_alignment_ball_only_mean_seconds: 0.006685072602704167
  body_view_selection_provisional_mean_seconds: 0.020631643192609773
  camera_alignment_dense_mean_seconds: 0.06231768260477111
  body_view_selection_dense_mean_seconds: 0.010341756988782435
  process_lifetime_peak_rss_bytes: 970997760
  caps_outputs_bitwise_equal: true
  cuda_initialized: false
  clips_over_1024: 1
  clips_over_2048: 0
artifacts:
  run_dir: knowledge/runs/run-frame-sampling-limit-20261117
  output_dir: outputs/tennis_scene/evaluate/frame-sampling-limit-20261117
  preserved_upstream: .worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/datasets/repeat-1/videos/video_000/clips/clip_035/annotations/tennis_scene
parents: []
relations: []
papers: []
tags: []
---

meiji_4camの69clipを処理するための上限調査を、#1748の親指示に基づき実施した。
1055 sampled frameでCPUの3段が動作することを確認し、既定の
`frame_sampling.max_frames`を1024から2048へ変更する。
標準pipelineの許容長を変えるため`summary_impact: updated`とした。
通常のGPU全段処理やC1のcold/warm基準値を完了したという記録ではない。

入力は保存済みclip_035（2108 frame、60000/1001 fps、4camera）の校正・ball・pose。
元datasetへは書かず、保存済みartifactsをread-onlyで復号した。
[入力manifest](../../runs/run-frame-sampling-limit-20261117/input-artifacts.json)は
97ファイルのSHA・元artifact identityを固定し、全ファイルを照合した。
scene.jsonのSHAは`7511de2f8d89edfbff0244fa5e33b4a946588581a8d18d218326da9478e2e437`。
全69clipのdataset.json SHAは
`968da7df0a235c4213af549ea661fa1b66c8e8c57c94b2c66d07775b23f50099`。
1024を超えるのはclip_035だけ（1055）、2048を超えるclipはなかった。
[全clip長・identity比較](../../runs/run-frame-sampling-limit-20261117/identity-and-lengths.json)。

1024では従来どおり`clip_too_long`で拒否する。2048/4096では同じ1055点を
使って完了し、各5反復・両上限で全出力fieldのhashが一致した。
別の再実行でも全配列のfiniteと元のhash一致を確認した。
court_sideは4cameraの半回転`[false,false,true,true]`を決定し、支持456 frame、
margin 0.23132368376254958、8仮説のcostは全てfiniteだった。

| CPU条件（上限2048） | 平均秒 | 標本標準偏差秒 | 観測の範囲 |
|---|---:|---:|---|
| court_side | 0.05531 | 0.00634 | 保存済み校正・ballの実観測 |
| camera_alignment | 0.00669 | 0.00257 | 実ballのみ。人物証拠は明示的に空 |
| body_view_selection | 0.02063 | 0.00245 | 保存済みpose/boxに暫定carrier-index IDを付与 |
| camera_alignment dense | 0.06232 | 0.00563 | 実校正へ投影した制御2人の全frame観測＋実ball |
| body_view_selection dense | 0.01034 | 0.00291 | 制御2人。各1055 frameのrequestを生成 |

人物対応は上流の旧停止点までに未生成だったため、実際のplayer_association結果を
使った検証ではない。暫定ID・dense入力は計算量と数値の確認に限る。
暫定IDのbody requestは4件が1054 frame、残る1件が311 frameだった。
実際の3D人物精度・GVHMRのGPUメモリ・全69clipの完走を保証しない。

process lifetime peak RSSは970,997,760 bytes（約926 MiB）。import・保存済み配列・
dense fixtureを含む同じprocessの最大で、段ごとの追加必要量ではない。
2msのRSS samplingも行ったが、短い割当を全て捕捉する保証はない。
`torch.cuda.is_initialized()`はfalse、GPU推論は実施していない。
実測値・全反復・環境は[results.json](../../runs/run-frame-sampling-limit-20261117/results.json)、
[finite/hashの再確認](../../runs/run-frame-sampling-limit-20261117/numerical-validation.json)、
[コマンドと再現条件](../../runs/run-frame-sampling-limit-20261117/reproduce.md)に保存した。

上限の由来はcommit `3879a27d7975c503ba1ba3f40f38948defdfe7f8`。
commit本文にはRe-ID inference policyのsampling budgetを移したとある。
[原文](../../runs/run-frame-sampling-limit-20261117/cap-history.txt)。
現行3段の固定tensor長や1024固有の数値制約を示す根拠は見つからなかった。
現行のsamplingは約30fpsのfull-clip gridを作り、超過を拒否する。
上限を上げても同じ入力のgridは変わらず、切捨て・密度・閾値・cameraは変更しない。
2048を超えるclipは引き続き拒否する。

同一Python・同一資産でstandard_definitionを組み立てるとsettingsが直接変わるのは
`court_side`、`camera_alignment`、`body_view_selection`。
依存先を含む9部品（この3件、player_association、player_triangulation、
ball_triangulation、gvhmr、body_placement、scene_assembly）はexecution identityが
変わり、Court/ball/person/poseの上流29部品は変わらない。
publicationのprocessing_settingsも変わるため、短いclipを含めdatasetのcompleted
markerは別identityになる。既存datasetを再生成する場合は`overwrite=true`が必要。
componentのexecuteは依存9部品を再計算し、旧identityの明示loadは拒否する。
このPRはsrcのYAMLだけを変え、Pythonの全体source fingerprintを変えない。
別revisionへ更新することによる全体fingerprintの変化はこのcap-only比較と別である。

テストはconfigurationの既存nodeへ1件追加し、2108 frameの具体的sample indexと
2048境界、2049点を切り詰めず拒否する契約を守る。
既存のgeometry testには明示1024での拒否検査があり、そのまま再利用した。
関連37件は9.28秒で成功（`pytest.log`）、collect-onlyは37件・6.09秒。
追加nodeは1件・call 0.08秒・setup 0.00秒だった。実GPU・通常pytestへの長いfixture追加はない。
現行mainの既定Courtはlegacyを保ち、#2191の動画共有はこの測定に含めていない。
従って上限修正だけで全69clipのCourt停止が解消するとは扱わない。
