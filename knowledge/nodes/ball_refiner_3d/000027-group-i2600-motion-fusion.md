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
  manifest_sha256: 790f05016e31ff4eb3fa8fc7d681d1a22ff0540928672e2a274a64543cd2a28a
  prepared_sha256: e0c54222f9cfbfde278956e4438d158fe9fa4a17e92ec791b607bac396c24677
  checkpoint_sha256: b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0
  event_threshold: 0.5
  event_min_gap_frames: 3
  tolerance_frames: 2
metrics:
  ball_only_test_macro_f1: 0.6687952320136229
  per_player_test_macro_f1: 0.6367107810098179
  set_attention_test_macro_f1: 0.6794645615735071
  shared_test_macro_f1: 0.672777869875931
repro:
  commit: 3d8cec367e8a9fb2cb44788bdae3eb494e0aeb3f
  command: bash knowledge/runs/group-i2600-motion-fusion/repro.sh "$REPORT_DIR"
artifacts:
  run_dir: knowledge/runs/group-i2600-motion-fusion
  output_dir: .worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training/coverage-r2
  curves: knowledge/runs/group-i2600-motion-fusion/learning-curves.png
members:
- run-i2600-motion-ball-only-s42-r2
- run-i2600-motion-shared-s42-r2
- run-i2600-motion-per-player-s42-r2
- run-i2600-motion-set-attention-s42-r2
- run-i2600-motion-shared-s43-r2
- run-i2600-motion-per-player-s43-r2
- run-i2600-motion-set-attention-s43-r2
- run-i2600-motion-shared-s44-r2
- run-i2600-motion-per-player-s44-r2
- run-i2600-motion-set-attention-s44-r2
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

Set Attentionは明治v3注釈の全イベントとポイント内で最も高い平均macro-F1を得た。独立ラベル6本で球のみを上回ったのはSharedとPer Playerで、Sharedの平均が最も高かった。Set Attentionを明治v3での追加検証候補とし、既定モデルの置換は保留する。

## 比較

明治4カメラの元フレーム軸を使い、47 train・11 val・11 testの固定分割で比較した。3方式は球embedding・時間trunk・両headを固定し、同じ161次元の選手特徴、AdamW、12 epoch・564更新、seed 42・43・44を使った。球のみは同じ事前学習checkpointを固定testで推論した。ピーク閾値0.5、同クラス間隔3 frame、元フレーム±2の1対1照合は全方式で共通。valのmacro-F1、同点時はBCE、さらに同点時は早いepochで重みを選んだ。 [分割・ラベル・sceneのハッシュ](../../runs/group-i2600-motion-fusion/dataset.json) · [実行条件とジョブID](../../runs/group-i2600-motion-fusion/pins.json)

判定不能区間を含む全てのcertain・probableからGaussian教師（σ=2 frame）を作った。BCEはcoverage=falseのフレームとクラス別のuncertain前後2 frameを除いた。採点は既知正例をcoverageにかかわらず照合し、判定不能区間の未照合予測をFP集計から除いた。この条件に当てはまる正例はtrain 10、val 4、test 1件だった。 [固定入力と正例件数の照合](../../runs/group-i2600-motion-fusion/verified-inputs.json)

| 方式 | v3 全イベント | v3 ポイント内 | 独立 全イベント | 独立 ポイント内 | 追加パラメータ | 中央クリップ ms | CUDAピーク MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 球のみ | 0.6688 | 0.7259 | 0.5011 | 0.5785 | 0 | 6.38 | 79.33 |
| Shared | 0.6728 ± 0.0256 | 0.7326 ± 0.0294 | 0.5696 ± 0.0886 | 0.6557 ± 0.1086 | 53,888 | 6.53 | 117.29 |
| Per Player | 0.6367 ± 0.0669 | 0.6903 ± 0.0699 | 0.5060 ± 0.0018 | 0.5980 ± 0.0016 | 215,552 | 7.17 | 117.91 |
| Set Attention | 0.6795 ± 0.0058 | 0.7508 ± 0.0100 | 0.4868 ± 0.0130 | 0.5666 ± 0.0100 | 779,136 | 7.82 | 150.85 |

macro-F1は3 seedの平均と標本標準偏差。球のみは1回。計時は同じRTX 5060 Ti・torch 2.13.0+cu130で、各クリップ1回warmup、同期付きforward 3回の中央値を取り、11本の中央値をseed間で平均した。メモリは全test・seedでCUDA allocatorが記録した最大割当。Set Attentionは球のみ比で時間約22%、割当約90%増えた。 [集計JSON](../../runs/group-i2600-motion-fusion/comparison.json) · [全指標CSV](../../runs/group-i2600-motion-fusion/metrics.csv)

| 方式 | 打球F1（66件） | バウンドF1（60件） | ネットF1（5件） | 選択epoch（seed 42 / 43 / 44） |
| --- | ---: | ---: | ---: | --- |
| 球のみ | 0.7593 | 0.9138 | 0.3333 | N/A |
| Shared | 0.8182 | 0.8826 | 0.3175 | 5 / 7 / 6 |
| Per Player | 0.8359 | 0.8678 | 0.2063 | 3 / 2 / 9 |
| Set Attention | 0.8319 | 0.9049 | 0.3016 | 1 / 1 / 8 |

v3では追加学習の9 runすべてで打球F1が上がった（球のみ0.7593、追加学習0.7797〜0.8667）。Set Attentionの打球は再現率0.621→0.737、適合率0.976→0.954、照合時刻MAE 15.46→14.05 ms。バウンド・ネットの低下が打球の改善を一部相殺し、Per Playerはseed 43でネット5件を全件見逃した。 [学習曲線](../../runs/group-i2600-motion-fusion/learning-curves.png)

valのmacro-F1は9 run全体で0.462〜0.723で、同じrun内でもepoch間の最大値と最小値に0.113〜0.231の差があった。valは11クリップ・ネット接触6件で、選択epochは1〜9に分かれた。球のみとのv3全イベントmacro-F1差は、Shared +0.0040、Per Player −0.0321、Set Attention +0.0107。SharedとPer Playerの差の絶対値はそれぞれのseed間標準偏差（0.0256、0.0669）より小さかった。Set Attentionのseed間標準偏差は0.0058で、ポイント内の平均差は+0.0249だった。 [各runのvalスコア範囲と選択epoch](../../runs/group-i2600-motion-fusion/comparison.json)

独立ラベル6本の採点対象は打球39・バウンド45・ネット4件。Sharedの平均0.5696にはseed 43の0.6719が効いている。seed 43の `clip_064` のネット候補f1184は、独立ラベルf1184でTP、v3注釈f1181では3フレーム差でFPだった。各ラベル源に同じ時刻許容を使った。ポイント内のFPは注釈イベントから推定した区間で数えた。独立ラベルではSet Attentionの全イベント平均は球のみより0.0143低かった。

修正前後で、v3全イベントの球のみとの差はSharedが−0.0057から+0.0040、Set Attentionが+0.0047から+0.0107に変わった。

## 条件別と代表例

- 球欠測および全選手SMPL欠測のフレームには、両ラベル源で採点対象イベントが0件だった。該当クラスの指標はN/A。独立ラベルでは球のみとSet Attentionのseed 42が球欠測フレームでネット候補1件ずつをFPとして計上した。
- v3で一部選手のSMPLが欠測した打球6件では、球のみ4件、SharedとSet Attentionは4〜5件、Per Playerは各5件が一致した。同条件のバウンド5件は全runで一致した。観測球がネット付近（`abs(y) ≤ 1 m`）のネット接触5件では、球のみ・Shared・Set Attentionが各1件、Per Playerはseed順に1・0・1件が一致した。
- v3のボレー8件は、球のみ5件、Shared 6・6・6件、Per Player 6・4・5件、Set Attention 5・5・5件が一致した。 `clip_040` f230のボレーは球のみが見逃し、Sharedの全seedとSet Attentionの2 seedが一致した。 `clip_048` f173のバウンドは球のみ、Per Playerのseed 42、Set Attentionのseed 42がf171で一致した。 `clip_056` f681のネット接触と、判定不能区間にある `clip_048` f367の有力打球は全runが見逃した。図の線はseed平均、色の帯は最小〜最大、灰色の破線は閾値0.5、青灰色の縦帯は許容±2 frame。 [元フレーム上の予測例](../../runs/group-i2600-motion-fusion/event-examples.png)

ネット接触と球欠測時の正例の少なさが改善を判断する上での制約だった。独立ラベルで欠測時やネット付近の改善を判断するには、該当する正例を含む追加クリップが必要になる。全層fine-tuningは今回の比較では見送った。

## 保存と再現

- 実験の完全HEADは `3d8cec367e8a9fb2cb44788bdae3eb494e0aeb3f`。学習済み融合重み9本、test予測110ファイル、照合JSON、設定・全epochの曲線をメインcheckout基準の `.worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training/coverage-r2/` に保存した。初回の予測・重みは親ディレクトリ下の初回runに保存している。 [成果物ごとのSHA-256](../../runs/group-i2600-motion-fusion/run-evidence.json)
- 前処理済み `inputs.json` のSHA-256は `e0c54222f9cfbfde278956e4438d158fe9fa4a17e92ec791b607bac396c24677`。最終配布版v3 manifestのSHA-256は `790f05016e31ff4eb3fa8fc7d681d1a22ff0540928672e2a274a64543cd2a28a`。3方式の事前学習checkpointは `b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0`。学習側の `clip_057` で初期座標・logitsのビット一致と手首192点の単位・向き・maskを最終配布版で確認した。 [実データ手首検証](../../runs/group-i2600-motion-fusion/clip_057_verification.json)
- 集計と図を保存済み予測から再生成するには、`MOTION_IMPLEMENTATION` に上記HEADの実装checkoutを指定する。既定は同じリポジトリの担当worktree。値は絶対パスを使い、`REPORT_DIR` は新規出力先とする。 `MOTION_RUN_ROOT` は保存済みrunのルートを差し替えるときに指定する。 [集計・可視化スクリプト](../../runs/group-i2600-motion-fusion/compare.py)

```bash
bash knowledge/runs/group-i2600-motion-fusion/repro.sh "$REPORT_DIR"
```

再学習には実装checkoutの Ball Refiner 3D README の `post_train prepare` と `post_train run` を使用する。`prepare` は公開v3、ライセンス取得済みSMPL-X、#2542のpinと独立6本を受け取る。`run` は各ノードに保存したqueueの `run.json` の引数を新規出力先で再現し、同じ3 seed・3方式と球のみ参照を共有training queueに投入する。 [実装PR #2613](https://github.com/Motoki0705/tennis-lab/pull/2613)
