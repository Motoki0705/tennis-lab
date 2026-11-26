---
id: run-i2542-meiji-ball-events
type: run
task: tennis_scene
sequence: 96
recorded_at: '2026-11-26'
title: 明治4カメラの3Dボールイベント検出比較
issue: 2542
provider: codex
date: '2026-11-26'
status: done
summary_impact: updated
config:
  benchmark: meiji_4cam/multiview-v2
  fps: 59.94005994005994
  clips: 69
  tolerance_frames: [2, 5, 10]
  primary_tolerance_frames: 5
  refiner_peak_threshold: 0.5
  refiner_peak_min_separation_frames: 3
  seed: 42
metrics:
  refiner3_shot_f1: 0.8484848484848485
  refiner3_bounce_f1: 0.8311688311688312
  refiner3_net_f1: 0.0
  rule_shot_f1: 0.868421052631579
  rule_bounce_f1: 0.7142857142857143
  rule_net_f1: 0.75
  refiner3_ok_clips: 69
  rule_ok_clips: 69
repro:
  commit: e4438667dd41162c6f4972a92146eb9b59bbba54
  command: bash knowledge/runs/run-i2542-meiji-ball-events/repro.sh
artifacts:
  run_dir: knowledge/runs/run-i2542-meiji-ball-events
  output_dir: .worktree-outputs/parent-1722-issue-2542-primary/tennis_scene/evaluate/meiji-ball-events/multiview-v2-20261126
  log: ~/.local/state/tennis-lab-agents/parent-sessions/local-jobs/20261126T004424Z_1d77d35bf621_meiji-ball-events/output.log
tags: [meiji_4cam, ball_events]
---

## 判定

明治4カメラの69本を両方式で処理できた。採点6本では、打球は規則、バウンドは3クラスRefinerのF1が高かった。ネット接触は規則で4件中3件を±5フレームで拾い、残る1件は9フレーム遅れた。Refinerは採点用の4件をすべて見逃した。規則のネット候補は映像確認に使えるが、参照4件では自動確定の精度を判断できない。打球・バウンドともポイント外の候補に抜けが多い。

旧 `traj3d_transformer` の学習済み重みは、親が確認したローカルD:/E:の `outputs`・`ckpt` とDriveの深さ5までの `*event*` 検索で回収できなかった。互換重みと学習・前処理設定を回収できた時点で、旧入力 `ball_pos_world` の座標・FPS・欠測・有効区間・系列長を照合して旧モデルの評価を再開できる。2026-11-26の判断で今回は3クラスRefinerを測った。 [旧コード dc8ec444](https://github.com/Motoki0705/tennis-lab/tree/dc8ec44457208d34b7e05fec0292add59223d8c8/src/tasks/event_detection) · [親の代替判断](https://github.com/Motoki0705/tennis-lab/issues/2506#issuecomment-7169560106) · [Refiner #2571](https://github.com/Motoki0705/tennis-lab/pull/2571) · [規則 #2569](https://github.com/Motoki0705/tennis-lab/pull/2569)

## 入力と照合

- 調整用は `clip_001`・`009`、採点用は `clip_027`・`028`・`040`・`043`・`056`・`064`。4視点RGBの独立ラベルを正解とした。Refinerは合成testで使った全クラス閾値0.5、ピーク最小間隔3フレームを採用した。映像ラベルによる閾値や時刻の補正は適用していない。 [選定記録](https://github.com/Motoki0705/tennis-lab/issues/2542#issuecomment-7170269245) · [入力ハッシュ](../../runs/run-i2542-meiji-ball-events/inputs.json)
- Refinerは `RefinerPredictor.from_checkpoint()` と `predict_scene_events()` を使用した。SceneResult v2のコート座標mと3D有効maskを使い、欠測の入力値は0で埋めてmaskを渡す。checkpointの60 fpsと入力の差約0.05994 fpsを許容0.06 fpsで受け入れた。seed 42、CPU 4 threads、`torch 2.13.0+cu130`。全長を使い、最長2108フレームで、1024を超えるクリップは11本だった。 [Refinerの設定](../../runs/run-i2542-meiji-ball-events/refiner-config.json)
- 両方式のフレームはSceneResultと元動画で共通の0始まり。秒は `frame / (60000 / 1001)`。予測JSONにはclip ID、元フレーム、種別、score、方式とバージョン、sceneのSHA-256がある。規則出力は実行コミット `aac549c1181e8e34105a00d54fabe8f93f8ed700` の既存出力を使用した。
- クリップと種別ごとに許容内の照合件数を最大化し、その中で絶対フレーム差の合計を最小にする一対一照合を使った。`certain`・`probable` を対象にし、判定不能区間内のラベル・予測は除外した。`uncertain` は正解から外し、他の確定ラベルと一致しなかった予測のうち同種の `uncertain` に一対一で対応するものは別計上した。ポイント前後の打球・バウンドも対象にした。ネット通過のみはnetの対象外。
- 採点用の除外は `uncertain` 2件（028 f501 bounce、040 f246 net）、判定不能4区間267フレーム、区間内の確定ラベル1件（028 f511、別球の不可視区間と重なったサーブ）、区間内の予測は各方式2件。±5で `uncertain` と対応して別計上された予測は0件。調整・採点別、種別ごとの母数と除外はJSONに保存した。 [集計JSON](../../runs/run-i2542-meiji-ball-events/metrics.json)

## 採点6本

主集計は±5フレーム（83.4 ms）。中央値と平均絶対誤差は一致ペアの絶対時刻差。符号付き中央値と許容別の全指標も集計JSONに載せた。

| 方式 | 種別 | 対象数 | TP / FP / FN | 適合率 | 再現率 | F1 | F1 ±2 | F1 ±10 | 絶対誤差中央値 / 平均 (ms) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Refiner | 打球 | 38 | 28 / 0 / 10 | 1.000 | 0.737 | 0.848 | 0.788 | 0.848 | 16.7 / 20.9 |
| Refiner | バウンド | 45 | 32 / 0 / 13 | 1.000 | 0.711 | 0.831 | 0.727 | 0.831 | 16.7 / 21.9 |
| Refiner | ネット接触 | 4 | 0 / 1 / 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| 規則 | 打球 | 38 | 33 / 5 / 5 | 0.868 | 0.868 | 0.868 | 0.737 | 0.868 | 16.7 / 21.7 |
| 規則 | バウンド | 45 | 25 / 0 / 20 | 1.000 | 0.556 | 0.714 | 0.629 | 0.714 | 16.7 / 16.7 |
| 規則 | ネット接触 | 4 | 3 / 1 / 1 | 0.750 | 0.750 | 0.750 | 0.500 | 1.000 | 0.0 / 16.7 |

## 調整用2本

同じ設定と±5フレームで別集計した。

| 方式 | 種別 | 対象数 | TP / FP / FN | 適合率 | 再現率 | F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Refiner | 打球 | 17 | 8 / 0 / 9 | 1.000 | 0.471 | 0.640 |
| Refiner | バウンド | 14 | 11 / 0 / 3 | 1.000 | 0.786 | 0.880 |
| Refiner | ネット接触 | 1 | 1 / 0 / 0 | 1.000 | 1.000 | 1.000 |
| 規則 | 打球 | 17 | 14 / 2 / 3 | 0.875 | 0.824 | 0.848 |
| 規則 | バウンド | 14 | 12 / 0 / 2 | 1.000 | 0.857 | 0.923 |
| 規則 | ネット接触 | 1 | 0 / 0 / 1 | — | 0.000 | 0.000 |

Refinerのネット接触1件は `clip_009` の正解f105に対しf108（+50.1 ms）。合成testでは閾値0.5で打球・バウンド・ネット接触の順にF1が .932・.986・.857だった。 [合成testの記録 #2574](https://github.com/Motoki0705/tennis-lab/pull/2574)

## 誤検出と見逃し

| 現象 | 採点用の代表例 |
| --- | --- |
| ポイント前後の打球・バウンド | ポイント外のバウンドは23件中Refinerが10件、規則が7件。Refinerのバウンド見逃し13件がすべてポイント外だった。規則の打球見逃し5件もすべてポイント外。056のラケットで球を返すf262と手で止めるf323はいずれも両方式で未照合。 |
| 遮蔽 | 056 f579のバウンドは選手との重なりがあり、規則が見逃した。再構築の球高は約0.13 m。遮蔽と球高が同時に見られ、どちらの寄与かは未同定。 |
| ボレー | 028 f689の低いボレーはRefiner f691・規則 f690で一致した。Refinerは040 f229と028 f1056の別のボレー2件を見逃した。 |
| ネット付近 | 規則は043 f119→f122、056 f681→f681、064 f1184→f1184。028ではポイント外の白帯接触f254に対してf263で、±5ではFPとFNが各1件、±10では同じ1件が照合された。 |
| 軌跡の外れと欠測 | 規則が発報した028 f323と056 f343付近では隣接3D点から約317 m/s・183 m/sの速度が出ており、再構築外れの疑いがある。064は1252フレーム中270が3D欠測で、Refinerの唯一の誤ネットf440は3D欠測上だった。規則の余分な打球f418も欠測の近傍だった。 |
| 種別の混同 | 040で規則の打球f401は、映像のバウンドf403の近傍にあった。 |

例の文脈には重複がある。全FP・FNの文脈別件数は集計JSONの `mismatch_context_counts_at_5_frames`、元フレームと近傍予測は出力先の `matches-at-5-frames.json` に収録した。

`clip_040` は4視点ともPTS 0から1001刻み（time base 1/60000）だった。RGB根拠画像で打球は視点間で3フレーム、バウンドは4フレームの幅を確認した。接触点の見え方も影響するため、固定カメラオフセットは特定できなかった。この観測幅は50.1〜66.7 msで、主許容は83.4 ms。 [同期確認](../../runs/run-i2542-meiji-ball-events/sync-check.json)

## 全69本と未ラベル61本

両方式とも69成功、失敗・未処理0。全69本の候補数は打球・バウンド・ネット接触の順にRefiner **259 / 371 / 21**、規則 **425 / 363 / 35**。 `clip_064` は両方式で完走し、3D有効982/1252を処理状況に記録した。 [処理状況](../../runs/run-i2542-meiji-ball-events/processing.json)

未ラベル61本は同じクラス・クリップ内で一対一照合し、一致率を2×ペア数÷両方式の候補数合計とした。

| 種別 | 候補数 Refiner / 規則 | クリップ別中央値 Refiner / 規則 | p90 Refiner / 規則 | ±5のペア数 | 一致率 ±2 / ±5 / ±10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 打球 | 221 / 369 | 3 / 5 | 7 / 11 | 210 | .664 / .712 / .715 |
| バウンド | 328 / 326 | 5 / 4 | 9 / 10 | 296 | .905 / .905 / .905 |
| ネット接触 | 19 / 31 | 0 / 0 | 1 / 1 | 19 | .600 / .760 / .760 |

両方式ともネット候補0件のクリップは35/61。精度は独立ラベルのある8本で測定した。クリップ別件数とヒストグラムも保存した。 [一致率と件数分布](../../runs/run-i2542-meiji-ball-events/unlabeled-agreement.json)

## 再現

メインcheckoutからのパスを以下に記す。再現コマンドは専用worktreeのrepo rootから実行する。

| 入力または出力 | 保管パス |
| --- | --- |
| benchmark | `data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v2` |
| checkpoint | `.worktree-outputs/parent-1722-issue-2567-primary/ball_refiner_3d/train/regression/net-contacts-s42-v1/logs/version_0/checkpoints/best.ckpt` |
| 規則出力 | `.worktree-outputs/parent-1722-issue-2568-primary/tennis_scene/analyze/rule_ball_events/meiji-multiview-v2-20261126` |
| 選定 | `.worktree-outputs/parent-1722-issue-2542-primary/tennis_scene/analyze/event-eval-selection/20261125T175807Z-e5e170/selection.json` |
| 目視ラベル | `.worktree-outputs/parent-1722-issue-2542-label-<番号>/tennis_scene/analyze/event-eval-labels/clip_<番号>/labels.json` |
| Refiner・照合の全出力 | `.worktree-outputs/parent-1722-issue-2542-primary/tennis_scene/evaluate/meiji-ball-events/multiview-v2-20261126` |

Refinerのcheckoutは `e4438667dd41162c6f4972a92146eb9b59bbba54` をcleanな状態で用意する。別のcheckoutや保管先は `MEIJI_REFINER_CHECKOUT`・`MEIJI_BENCHMARK`・`MEIJI_REFINER_CHECKPOINT`・`MEIJI_SELECTION`・`MEIJI_RULE_RUN`・`MEIJI_LABEL_ROOT` で指定できる。

```bash
bash knowledge/runs/run-i2542-meiji-ball-events/repro.sh
.venv/bin/python -B knowledge/runs/run-i2542-meiji-ball-events/check_protocol.py
```

省略時は `outputs/tennis_scene/evaluate/meiji-ball-events/multiview-v2-20261126` に出力する。既存のRefiner JSONがあれば来歴とフレーム格子を照合して再利用する。最初から推論するときは、空の出力ディレクトリを絶対パスで第1引数に渡す。8本の推論を先行し、残り61本と全集計のローカルジョブは174.285秒で完了した。重み・manifest・選定・ラベル・既存規則出力の固定値、全138予測の受領時ハッシュ、照合出力のハッシュを同梱した。 [入力の固定値](../../runs/run-i2542-meiji-ball-events/inputs.json) · [全予測とprogressのハッシュ](../../runs/run-i2542-meiji-ball-events/artifact-hashes.json) · [照合出力のハッシュ](../../runs/run-i2542-meiji-ball-events/output-sha256.txt)
