---
id: group-i2600-motion-fusion
type: group
task: ball_refiner_3d
sequence: 27
recorded_at: '2026-11-26'
title: 明治4カメラでのSMPLモーション融合3方式の比較
issue: 2600
summary_impact: updated
config:
  dataset: meiji_4cam/multiview-v3
  split_counts:
  - 47
  - 11
  - 11
  seeds:
  - 42
  - 43
  - 44
  epochs: 12
  updates_per_training_run: 564
  frozen_baseline: true
  manifest_sha256: e456f07c510b25ef4730336b18ae850090146c15cb1407c78d120ca53f4edea3
  prepared_sha256: 2d4b16b9a2f38d7097759b763ae3b7c81018233168afffef06a2c06a7bdf2bd4
  checkpoint_sha256: b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0
  event_threshold: 0.5
  event_min_gap_frames: 3
  tolerance_frames: 2
metrics:
  ball_only_test_macro_f1: 0.671160525656175
  per_player_test_macro_f1: 0.6054309436796605
  set_attention_test_macro_f1: 0.6758947168038836
  shared_test_macro_f1: 0.6654843610659613
repro:
  commit: 976a71056e562a19f6a646c8004c4f081282246a
  command: bash knowledge/runs/group-i2600-motion-fusion/repro.sh "$REPORT_DIR"
artifacts:
  run_dir: knowledge/runs/group-i2600-motion-fusion
  output_dir: .worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training
  curves: knowledge/runs/group-i2600-motion-fusion/learning-curves.png
members:
- run-i2600-motion-ball-only-s42
- run-i2600-motion-shared-s42
- run-i2600-motion-per-player-s42
- run-i2600-motion-set-attention-s42
- run-i2600-motion-shared-s43
- run-i2600-motion-per-player-s43
- run-i2600-motion-set-attention-s43
- run-i2600-motion-shared-s44
- run-i2600-motion-per-player-s44
- run-i2600-motion-set-attention-s44
parents:
- run-i2567-net-contacts-s42
relations:
- to: run-i2542-meiji-ball-events
  rel: related
  note: 固定test内の採点用6本を独立ラベルで再採点した。
papers: []
tags:
- meiji_4cam
- ablation
- ball_events
- smpl_motion
---

Set Attentionは明治v3注釈の全イベントで追加学習3方式の平均macro-F1が最も高かった。球のみとの差は **+0.0047**、ポイント内は **+0.0234** だった。独立6本では差が **−0.0018** で、球のみを上回る再現性は確認できなかった。採用候補はSet Attentionとし、置換の判断は保留する。

## 比較

明治4カメラの元フレーム軸を使い、47 train・11 val・11 testの固定分割で比較した。3方式は球embedding・時間trunk・両headを固定し、同じ161次元の選手特徴、AdamW、12 epoch・564更新、seed 42・43・44を使った。球のみは同じ事前学習checkpointを追加更新せず評価した。certain・probableをGaussian σ=2 frameの教師にし、判定不能とクラス別uncertain前後2 frameをlossから除いた。ピーク閾値0.5、同クラス間隔3 frame、元フレーム±2の1対1照合は全方式で共通。valのmacro-F1、同点時はBCE、さらに同点時は早いepochで重みを選んだ。 [分割・ラベル・sceneのハッシュ](../../runs/group-i2600-motion-fusion/dataset.json) · [実行条件とジョブID](../../runs/group-i2600-motion-fusion/pins.json)

| 方式 | v3 全イベント | v3 ポイント内 | 独立 全イベント | 独立 ポイント内 | 追加パラメータ | 中央クリップ ms | CUDAピーク MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 球のみ | 0.6712 | 0.7259 | 0.5051 | 0.5839 | 0 | 7.21 | 79.33 |
| Shared | 0.6655 ± 0.0215 | 0.7222 ± 0.0247 | 0.5656 ± 0.0905 | 0.6453 ± 0.1164 | 53,888 | 7.04 | 117.29 |
| Per Player | 0.6054 ± 0.0775 | 0.6551 ± 0.0846 | 0.4975 ± 0.0164 | 0.5866 ± 0.0135 | 215,552 | 7.57 | 117.91 |
| Set Attention | 0.6759 ± 0.0105 | 0.7493 ± 0.0088 | 0.5033 ± 0.0153 | 0.5751 ± 0.0149 | 779,136 | 9.01 | 150.85 |

macro-F1は3 seedの平均と標本標準偏差。球のみは1回。計時は同じRTX 5060 Ti・torch 2.13.0+cu130で、各クリップ1回warmup、同期付きforward3回の中央値を取り、11本の中央値をseed間で平均した。メモリは全test・seedのCUDA allocator最大割当。Set Attentionは球のみ比で時間約25%、割当約90%増えた。 [集計JSON](../../runs/group-i2600-motion-fusion/comparison.json) · [全指標CSV](../../runs/group-i2600-motion-fusion/metrics.csv)

| 方式 | 打球F1（65件） | バウンドF1（60件） | ネットF1（5件） | 選択epoch（seed 42 / 43 / 44） |
| --- | ---: | ---: | ---: | --- |
| 球のみ | 0.7664 | 0.9138 | 0.3333 | N/A |
| Shared | 0.8194 | 0.8596 | 0.3175 | 3 / 7 / 11 |
| Per Player | 0.8308 | 0.8744 | 0.1111 | 3 / 2 / 8 |
| Set Attention | 0.8264 | 0.8997 | 0.3016 | 10 / 2 / 9 |

3方式ともv3の全seedで打球F1が上がった。Set Attentionの打球は再現率0.631→0.733、適合率0.976→0.948、照合時刻MAE 15.46→13.15 ms。バウンド・ネットの低下が打球の改善を相殺した。Per Playerは2 seedでネット接触5件を全件見逃し、重みを独立にした効果は安定しなかった。Set Attentionはv3の全イベント・ポイント内で追加学習3方式中の最高平均と最小のseed間標準偏差だった。 [学習曲線](../../runs/group-i2600-motion-fusion/learning-curves.png)

valのmacro-F1は同じrunの中でもepoch間で0.14〜0.30動き（9 run全体で0.457〜0.781）、選択epochはseedにより2〜11に分かれた。val 11クリップ（ネット接触6件）では最良epochの選択がノイズに敏感で、球のみとのtest macro-F1差（Shared −0.006、Per Player −0.066、Set Attention +0.005）はいずれもseed間の標準偏差の範囲にある。全9 runで一貫した差は打球F1の改善だけだった。

独立ラベル6本の採点対象は打球38・バウンド45・ネット4件。Sharedの平均macro-F1 0.566にはseed 43の0.670が効いている。seed 43の `clip_064` ネット候補f1184は独立ラベルf1184でTP、v3注釈f1181では3フレーム差でFPだった。各ラベル源の結果を併記し、時刻許容は固定した。ポイント内のFPは注釈イベントから推定した区間で数えた。

## 条件別と代表例

- 球欠測および全選手SMPL欠測のフレームには、両ラベル源で採点対象イベントが0件だった。指標はN/A。独立ラベルで球のみが球欠測フレームに出したネット候補1件はFPとして集計した。
- v3で一部選手のSMPLが欠測した打球は6件。球のみ4件、SharedとPer Playerは4–5件、Set Attentionは4–5件が一致した。同条件のバウンド5件は全seedが一致した。ネット付近（観測球の `abs(y) ≤ 1 m`）でv3のネット5件に対し、球のみ・Shared・Set Attentionは各1件。Per Playerは1・0・0件だった。
- v3のボレー8件は、球のみ5件、Shared 6・6・6件、Per Player 6・4・5件、Set Attention 6・5・6件が一致した。 `clip_040` f230はSet Attention全seedが拾った。一方、`clip_048` f173のバウンドは球のみがf171で一致し、Set Attention全seedのピークは許容範囲より早かった。 `clip_056` f681のネット接触は全runが見逃した。図の線はseed平均、色の帯は最小–最大、灰色の破線は閾値0.5、青灰色の縦帯は許容±2 frame。 [元フレーム上の予測例](../../runs/group-i2600-motion-fusion/event-examples.png)

Set Attentionを追加検証の候補とする根拠は、v3で平均macro-F1が最も高く、打球の改善が全seedで出たこと。独立ラベルでは再現せず、ネットは5件、球欠測時の正例は0件だった。全層fine-tuningは今回の比較からは見送った。独立した追加クリップでネット接触と欠測時の正例を増やしてから採否を判断できる。

## 保存と再現

- 実験の完全HEADは `976a71056e562a19f6a646c8004c4f081282246a`。学習済み融合重み9本、test予測110ファイル、照合JSON、設定・全epochの曲線をメインcheckout基準の `.worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training/` に保存した。 [成果物ごとのSHA-256](../../runs/group-i2600-motion-fusion/run-evidence.json)
- 前処理済み `inputs.json` のSHA-256は `2d4b16b9a2f38d7097759b763ae3b7c81018233168afffef06a2c06a7bdf2bd4`。v3 manifestのSHA-256は `e456f07c510b25ef4730336b18ae850090146c15cb1407c78d120ca53f4edea3`。3方式の事前学習checkpointは `b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0`。CPU前処理の照合結果と学習側clipの手首の単位・向き・mask検証を保存した。 [入力照合](../../runs/group-i2600-motion-fusion/verified-inputs.json) · [実データ手首検証](../../runs/group-i2600-motion-fusion/clip_057_verification.json)
- 集計と図を保存済み予測から再生成するには、`MOTION_IMPLEMENTATION` に上記HEADの実装checkoutを指定する。既定は同じリポジトリの担当worktree。値は絶対パスを使い、`REPORT_DIR` は新規出力先とする。 `MOTION_RUN_ROOT` は保存済みrunのルートを差し替えるときに指定する。 [集計・可視化スクリプト](../../runs/group-i2600-motion-fusion/compare.py)

```bash
bash knowledge/runs/group-i2600-motion-fusion/repro.sh "$REPORT_DIR"
```

再学習では、実装checkoutの Ball Refiner 3D README の `post_train prepare` と `post_train run` を使用する。`prepare` は公開v3、ライセンス取得済みSMPL-X、#2542のpinと独立6本を受け取る。`run` は各ノードの queue 捕捉 `run.json` にある引数を新規出力先で再現し、同じ3 seed・3方式と球のみ参照を共有 training queue に投入する。 [実装PR #2613](https://github.com/Motoki0705/tennis-lab/pull/2613)
